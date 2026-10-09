#!/usr/bin/env python3
"""리쥬네브 릴스 엔진 v2 — 풀스크린, 형식·음악을 매번 바꿀 수 있음.
사용법: python3 reel_engine.py config.json

config 예시:
{
 "out": "/home/claude/out.mp4",
 "logo": "로고 이미지 경로(선택)",
 "transition": "stack|fade|whip|flash",
 "text": "bottom|center|top",
 "music": "lofi|upbeat|chill|acoustic|none",
 "scenes": [
   {"src": "a.jpg", "focus": 0.4, "label": "WEAR", "head": "가볍게 메는 미니 백팩", "sub": "단 230g", "beats": 4},
   {"src": "b.jpg", "focus": "contain", "light": true, "label": "SIZE", "head": "...", "sub": "..."}
 ],
 "end": {"src": "c.jpg", "focus": 0.4, "en": "JOE  COOL  BACKPACK", "kr": "조쿨백팩",
         "cta": "프로필 링크에서 구매  →", "url": "le-junev.com", "beats": 6}
}
- focus: 0~1(세로 어디를 보여줄지) 또는 "contain"(흰 배경 상품컷·컬러표: 자동으로 여백 잘라 크게)
- light: 흰 배경 사진이면 true → 글씨가 어두운색, 그림자 없음
- beats: 장면 길이를 음악 박자 수로 지정(없으면 bpm에 맞춰 약 2초). 장면 전환이 박자에 딱 맞음
"""
import json, sys, subprocess, tempfile, wave, os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
D = "/usr/share/fonts/opentype/noto/"
def S(w, s): return ImageFont.truetype(D + f"NotoSansCJK-{w}.ttc", s, index=1)
INK = np.array([.07, .07, .07], np.float32); GREY = np.array([.45, .44, .42], np.float32)
SAGE = np.array([.42, .55, .47], np.float32); WHITE = np.array([1, 1, 1], np.float32)
GOLDL = np.array([.92, .85, .68], np.float32); LGREY = np.array([.88, .86, .82], np.float32)
MOODS = {  # bpm, 코드(루트 포함 주파수)
    "lofi":     (84,  [[174.61,220,261.63,329.63],[164.81,196,246.94,293.66],[146.83,174.61,220,261.63],[130.81,164.81,196,246.94]]),
    "upbeat":   (120, [[261.63,329.63,392.0],[196.0,246.94,293.66],[220.0,261.63,329.63],[174.61,220.0,261.63]]),
    "chill":    (104, [[220.0,261.63,329.63,392.0],[146.83,174.61,220.0,261.63],[196.0,246.94,293.66,349.23],[130.81,164.81,196.0,246.94]]),
    "acoustic": (92,  [[196.0,246.94,293.66],[146.83,185.0,220.0],[164.81,196.0,246.94],[130.81,164.81,196.0]]),
}

def cl(t): return min(1, max(0, t))
def ease(t): t = cl(t); return 1 - (1 - t) ** 3
def eexp(t): t = cl(t); return 1 if t >= 1 else 1 - 2 ** (-10 * t)
def eio(t): t = cl(t); return t * t * (3 - 2 * t)

def comp(fr, mask, color, x, y, alpha=1.0):
    if alpha <= 0.003: return
    h, w = mask.shape[:2]; x = int(round(x)); y = int(round(y))
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    if m.ndim == 2: m = m[..., None]
    c = color if np.ndim(color) == 1 else color[y0 - y:y1 - y, x0 - x:x1 - x]
    fr[y0:y1, x0:x1] = fr[y0:y1, x0:x1] * (1 - m * alpha) + c * m * alpha

def rmask(w, h, r):
    m = Image.new("L", (w * 2, h * 2), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], r * 2, fill=255)
    return np.asarray(m.resize((w, h), Image.LANCZOS), np.float32) / 255

def prep(path, focus=0.5):
    """사진을 화면(1.1배 여유) 크기로 준비. contain이면 여백 자동 제거 후 크게 배치."""
    im = Image.open(path).convert("RGB"); bw, bh = int(W * 1.1), int(H * 1.1)
    if focus == "contain":
        a = np.asarray(im).astype(np.int16)
        col = tuple(int(x) for x in np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3)]).mean(0))
        ys, xs = np.where(np.abs(a - np.array(col)).max(2) > 22)
        if len(ys):
            p = int(0.04 * max(im.size))
            im = im.crop((max(0, xs.min() - p), max(0, ys.min() - p), min(im.width, xs.max() + p), min(im.height, ys.max() + p)))
        boxw, boxh, top = bw * 0.86, bh * 0.56, bh * 0.10
        sc = min(boxw / im.width, boxh / im.height); r = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
        if sc > 1.3: r = r.filter(ImageFilter.UnsharpMask(2, 90, 2))
        c = Image.new("RGB", (bw, bh), col); c.paste(r, ((bw - r.width) // 2, int(top + (boxh - r.height) / 2))); return c
    sc = max(bw / im.width, bh / im.height); r = im.resize((int(im.width * sc) + 1, int(im.height * sc) + 1), Image.LANCZOS)
    if sc > 1.6: r = r.filter(ImageFilter.UnsharpMask(2, 90, 2))
    x = (r.width - bw) // 2; y = int((r.height - bh) * float(focus)); return r.crop((x, y, x + bw, y + bh))

def frame_of(big, z):
    """z(0~1)만큼 천천히 확대된 화면 한 장"""
    bw, bh = big.size; w = int(bw / 1.1 * (1.1 - 0.08 * z)); h = int(w * H / W); x = (bw - w) // 2; y = (bh - h) // 2
    return np.asarray(big.crop((x, y, x + w, y + h)).resize((W, H), Image.BILINEAR), np.float32) / 255

class Txt:
    def __init__(s, text, font, track=0):
        s.chars = []; x = 0; ch = int(font.size * 1.5); pad = 22
        for c in text:
            cw = font.getlength(c)
            if c != " ":
                im = Image.new("L", (int(cw) + pad * 2, ch + pad * 2), 0); ImageDraw.Draw(im).text((pad, pad), c, font=font, fill=255)
                s.chars.append((im, x - pad))
            x += cw + track
        s.w = x - track; s.h = ch; s.pad = pad
    def draw(s, fr, cy, color, t, stagger=0.018, dur=0.5, out=0.0, blur=7, rise=14, shadow=0.0):
        x0 = W / 2 - s.w / 2
        for i, (im, ox) in enumerate(s.chars):
            k = ease((t - i * stagger) / dur); a = k * (1 - eio(out))
            if a <= 0.003: continue
            b = float((1 - k) * blur + eio(out) * 4)
            mk = im.filter(ImageFilter.GaussianBlur(b)) if b > 0.4 else im
            m = np.asarray(mk, np.float32) / 255; y = cy - s.h / 2 - s.pad + (1 - k) * rise
            if shadow > 0:
                sh = np.asarray(mk.filter(ImageFilter.GaussianBlur(8)), np.float32) / 255
                comp(fr, sh, np.zeros(3, np.float32), x0 + ox + 2, y + 4, a * shadow)
            comp(fr, m, color, x0 + ox, y, a)

def logo_mask(path):
    if path and os.path.exists(path):
        src = np.asarray(Image.open(path).convert("L"), np.float32); m = np.clip((205 - src) / 160, 0, 1)
        ys, xs = np.where(m > 0.1); m = m[ys.min() - 3:ys.max() + 4, xs.min() - 3:xs.max() + 4]
        return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((230, int(m.shape[0] * 230 / m.shape[1])), Image.LANCZOS), np.float32) / 255
    f = ImageFont.truetype(D + "NotoSerifCJK-Bold.ttc", 34, index=1)
    im = Image.new("L", (300, 60), 0); x = 10
    for c in "LEJUNEV": ImageDraw.Draw(im).text((x, 5), c, font=f, fill=255); x += f.getlength(c) + 6
    return np.asarray(im.crop((0, 0, int(x) + 10, 60)), np.float32) / 255

# ---------------- 음악 (직접 합성 = 저작권 없는 오리지널 음원) ----------------
def pluck(freq, n, sr, decay=0.996):
    p = max(2, int(sr / freq)); buf = np.random.default_rng(int(freq)).uniform(-1, 1, p); out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]; buf[i % p] = decay * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out

def music(path, dur, cuts, mood):
    SR = 44100; N = int(SR * dur); out = np.zeros((N, 2)); bpm, chords = MOODS[mood]; beat = 60 / bpm; bar = beat * 4
    rng = np.random.default_rng(3)
    def add(i, sig, l=1.0, r=1.0):
        n = min(len(sig), N - i)
        if n > 0: out[i:i + n, 0] += sig[:n] * l; out[i:i + n, 1] += sig[:n] * r
    for b in range(int(dur / bar) + 1):
        s = int(b * bar * SR); n = min(N - s, int(bar * SR))
        if n <= 0: break
        tt = np.arange(n) / SR; ch = chords[b % 4]
        if mood == "acoustic":
            for k in range(8):  # 아르페지오
                f = ch[k % len(ch)] * (2 if k % 4 == 3 else 1); i = s + int(k * beat / 2 * SR)
                add(i, pluck(f, int(beat * 2 * SR), SR) * 0.22, 0.8 if k % 2 else 1.0, 1.0 if k % 2 else 0.8)
            continue
        env = np.clip(tt / (0.05 if mood == "upbeat" else 0.3), 0, 1) * np.clip((bar - tt) / 0.3, 0, 1)
        for f in ch:
            for dt in (-0.7, 0.7):
                v = np.sin(2 * np.pi * (f + dt) * tt) * 0.5 + np.sin(2 * np.pi * 2 * (f + dt) * tt) * (0.25 if mood == "upbeat" else 0.1)
                if mood == "upbeat": v *= 0.6 + 0.4 * (np.sin(2 * np.pi * (bpm / 60) * 2 * tt) > 0)  # 리듬 스탭
                add(s, v * env * 0.045, 1 if dt < 0 else 0.6, 0.6 if dt < 0 else 1)
        add(s, np.sin(2 * np.pi * ch[0] / 2 * tt) * env * 0.11)
    def kick(i, a):
        n = int(0.35 * SR); tt = np.arange(n) / SR; add(i, np.sin(2 * np.pi * (48 + 70 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9) * a)
    def hat(i, a, l=0.05):
        n = int(l * SR); h = rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 70); h = h - np.convolve(h, np.ones(4) / 4, "same"); add(i, h * a)
    def clap(i, a):
        n = int(0.18 * SR); c = rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 25); c = np.convolve(c, np.ones(6) / 6, "same"); add(i, c * a)
    for k in range(int(dur / beat) + 1):
        i = int(k * beat * SR); half = int((k + 0.5) * beat * SR)
        if mood == "lofi":
            if k % 2 == 0: kick(i, 0.33)
            else: clap(i, 0.10)
            hat(half, 0.04)
        elif mood == "upbeat":
            kick(i, 0.38); hat(half, 0.06, 0.08)
            if k % 2 == 1: clap(i, 0.16)
        elif mood == "chill":
            kick(i, 0.26); hat(half, 0.05)
            if k % 4 == 2: clap(i, 0.08)
        elif mood == "acoustic":
            hat(i, 0.025, 0.03); hat(half, 0.018, 0.03)
    for c in cuts:  # 장면 전환 효과음
        i = int(c * SR); n = int(0.9 * SR); tt = np.arange(n) / SR
        add(i, sum(a * np.sin(2 * np.pi * 1046.5 * m * tt) * np.exp(-tt * d) for m, a, d in [(1, 1, 4), (2, .4, 6)]) * 0.025)
    rev = out.copy()
    for d, g in [(0.09, .28), (0.17, .2), (0.29, .14)]: k = int(d * SR); rev[k:] += out[:-k] * g
    t = np.arange(N) / SR; out = rev * np.clip(t / 0.4, 0, 1)[:, None] * np.clip((dur - t) / 1.2, 0, 1)[:, None]
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.62
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype(np.int16).tobytes())

# ---------------- 본체 ----------------
def main(cfg):
    sc = cfg["scenes"]; e = cfg["end"]
    trans = cfg.get("transition", "stack"); tstyle = cfg.get("text", "bottom"); mood = cfg.get("music", "lofi")
    bpm = MOODS.get(mood, (96, None))[0]; beat = 60 / bpm
    TR = {"stack": 0.5, "fade": 0.6, "whip": 0.35, "flash": 0.12}[trans]
    durs = [s.get("beats", max(2, round(2.0 / beat))) * beat for s in sc]
    starts = [float(x) for x in np.cumsum([0] + durs[:-1])]; ENDS = float(sum(durs)); DUR = ENDS + e.get("beats", max(4, round(3.6 / beat))) * beat
    N = int(DUR * FPS)
    IMG = [prep(s["src"], s.get("focus", 0.5)) for s in sc]; HERO = prep(e["src"], e.get("focus", 0.4))
    # 글씨 스타일
    if tstyle == "center": hf, sf, lf, TY = S("Black", 92), S("Medium", 40), S("Medium", 26), 860
    elif tstyle == "top": hf, sf, lf, TY = S("Bold", 64), S("Regular", 34), S("Medium", 24), 330
    else: hf, sf, lf, TY = S("Bold", 60), S("Regular", 33), S("Medium", 24), 1330
    LAB = [Txt(f"{i + 1:02d}   {s.get('label', '')}", lf, 6) for i, s in enumerate(sc)]
    HEAD = [Txt(s["head"], hf, -1) for s in sc]; SUB = [Txt(s.get("sub", ""), sf, 0) for s in sc]
    gap = (92 if tstyle == "center" else 78)
    LOGO = logo_mask(cfg.get("logo")); lw = LOGO.shape[1]
    eEn = Txt(e.get("en", ""), S("Medium", 26), 5); eKr = Txt(e["kr"], S("Bold", 58), -1)
    eCta = Txt(e.get("cta", "프로필 링크에서 구매  →"), S("Medium", 34), 0); eUrl = Txt(e.get("url", "le-junev.com"), S("Light", 26), 3)
    BTN = rmask(560, 104, 52)
    yv = np.linspace(0, 1, H)[:, None, None].astype(np.float32)
    if tstyle == "bottom": SCR = np.clip((yv - 0.45) / 0.55, 0, 1) ** 1.1 * 0.78 + np.clip((0.16 - yv) / 0.16, 0, 1) * 0.35
    elif tstyle == "top": SCR = np.clip((0.42 - yv) / 0.42, 0, 1) ** 1.1 * 0.72 + np.clip((yv - 0.85) / 0.15, 0, 1) * 0.2
    else: SCR = 0.30 + np.exp(-((yv - 0.46) / 0.14) ** 2) * 0.25 + 0 * yv
    SCR = SCR.astype(np.float32)
    rng = np.random.default_rng(5)
    GR = [np.repeat(np.repeat(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), 2, 0), 2, 1)[..., None] * 0.006 for _ in range(6)]

    def place(fr, img, x, y, alpha=1.0, dark=0.0, scale=1.0):
        if scale != 1.0:
            w, h = int(W * scale), int(H * scale)
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32) / 255
            x += (W - w) / 2; y += (H - h) / 2
        comp(fr, np.ones(img.shape[:2], np.float32), img * (1 - dark), x, y, alpha)

    def draw_scene(fr, img, lt, first):
        """lt: 이 장면 시작 후 경과 시간"""
        k = 1.0 if first else cl(lt / TR)
        if trans == "stack": place(fr, img, 0, (1 - eexp(k)) * H * 0.75)
        elif trans == "fade": place(fr, img, 0, 0, alpha=eio(k))
        elif trans == "whip": place(fr, img, (1 - eexp(k)) * W, 0)
        else: place(fr, img, 0, 0, scale=1 + 0.06 * (1 - ease(lt / 0.4)))

    tmp = tempfile.mkdtemp(); vid = f"{tmp}/v.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "slow", vid], stdin=subprocess.PIPE)
    allst = starts + [ENDS]
    for f in range(N):
        t = f / FPS; fr = np.zeros((H, W, 3), np.float32)
        cur = max([i for i, s0 in enumerate(allst) if s0 <= t])
        # 이전 장면(전환 중일 때만) + 현재 장면
        for i in ([cur - 1] if cur > 0 and t - allst[cur] < TR else []) + [cur]:
            lt = t - allst[i]; img_src = HERO if i == len(sc) else IMG[i]
            span = (DUR - ENDS) if i == len(sc) else durs[i]
            img = frame_of(img_src, cl(lt / (span + TR)))
            if i == cur - 1:
                kk = cl((t - allst[cur]) / TR)
                if trans == "stack": place(fr, img, 0, -30 * eexp(kk), dark=0.25 * eexp(kk), scale=1 - 0.06 * eexp(kk))
                elif trans == "whip": place(fr, img, -eexp(kk) * W * 0.35, 0, dark=0.2 * kk)
                else: place(fr, img, 0, 0)
            else:
                draw_scene(fr, img, lt, i == 0)
        if trans == "flash" and cur > 0 and t - allst[cur] < 0.18:
            fr = fr * (1 - 0.6 * (1 - (t - allst[cur]) / 0.18)) + 0.6 * (1 - (t - allst[cur]) / 0.18)
        # 그림자(흰 배경 상품컷은 약하게)
        L_cur = sc[cur].get("light", False) if cur < len(sc) else False
        L_prv = sc[cur - 1].get("light", False) if cur > 0 else L_cur
        kk = eexp((t - allst[cur]) / max(TR, 0.3)); lw_ = (1 if L_cur else 0) * kk + (1 if L_prv else 0) * (1 - kk)
        if cur == len(sc): fr *= 1 - 0.35 * ease((t - ENDS - 0.3) / 0.8)
        fr *= 1 - SCR * (1 - lw_)
        comp(fr, LOGO, WHITE if lw_ < 0.5 else INK, W / 2 - lw / 2, 118, ease(t / 0.6))
        # 장면 글씨
        for i in range(len(sc)):
            lt = t - starts[i]; span = durs[i]
            if lt < 0 or lt > span + 0.05: continue
            out = cl((lt - (span - 0.22)) / 0.22); Lg = sc[i].get("light", False); shd = 0 if Lg else 0.35
            ty = 1330 if Lg else TY  # 흰 배경 상품컷은 가방 아래에 글씨
            if sc[i].get("label"): LAB[i].draw(fr, ty, SAGE if Lg else GOLDL, lt - 0.12, out=out, blur=4, rise=8, shadow=shd)
            HEAD[i].draw(fr, ty + gap, INK if Lg else WHITE, lt - 0.18, out=out, shadow=shd)
            if sc[i].get("sub"): SUB[i].draw(fr, ty + gap * 2 - 8, GREY if Lg else LGREY, lt - 0.3, stagger=0.012, out=out, blur=5, rise=10, shadow=shd)
        if t >= ENDS:
            lt = t - ENDS; EY = 1330
            eEn.draw(fr, EY - 10, GOLDL, lt - 0.25, blur=4, rise=8); eKr.draw(fr, EY + 62, WHITE, lt - 0.32, shadow=0.3)
            bk = ease((lt - 0.7) / 0.5)
            if bk > 0:
                by = EY + 170 + (1 - bk) * 20; comp(fr, BTN, np.array([.98, .97, .95], np.float32), W / 2 - 280, by - 52, bk); eCta.draw(fr, by, INK, lt - 0.85, blur=3, rise=4)
            eUrl.draw(fr, EY + 270, LGREY, lt - 1.1, blur=3, rise=6)
        fr += GR[f % 6]
        p.stdin.write((np.clip(fr, 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    if mood != "none":
        music(f"{tmp}/a.wav", DUR, starts[1:] + [ENDS], mood)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-i", f"{tmp}/a.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", "-movflags", "+faststart", cfg["out"]], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-c", "copy", "-movflags", "+faststart", cfg["out"]], check=True)
    print("saved", cfg["out"], round(DUR, 1), "s", trans, tstyle, mood)

if __name__ == "__main__":
    main(json.load(open(sys.argv[1])))
