# PS5LM logo kit masters (direction A, "Grid"): PS5 over LM, the cursor in the cell under the 5.
# Letters: the site's Recursive Mono, outlined and merged with skia-pathops; no fonts needed to show the files.
# Usage: uv run --with fonttools --with brotli --with skia-pathops python brand/build.py brand
import sys, os
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from pathops import Path, op, PathOp

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'site', 'assets', 'fonts', 'recursive-mono-v1.woff2')
GROUND, INK_DARK, INK_LIGHT, ACCENT = '#0e0e14', '#f2f2f7', '#0e0e14', '#8288f0'
CAP, TRACK, LEAD = 700, -40, 0.3            # cap height, tracking, gap between the lines (in caps)
_fonts = {}

def glyphs(text, weight, x, base, track=TRACK):
    if weight not in _fonts:
        f = instantiateVariableFont(TTFont(FONT), {'wght': float(weight)})
        _fonts[weight] = (f.getGlyphSet(), f.getBestCmap())
    gs, cmap = _fonts[weight]
    out = []
    for i, ch in enumerate(text):
        p = Path()
        gs[cmap[ord(ch)]].draw(TransformPen(p.getPen(glyphSet=gs), (1, 0, 0, -1, x + i * (600 + track), base)))
        p.simplify()
        out.append(p)
    return out

def rect(x0, y0, x1, y1):
    p = Path(); pen = p.getPen()
    pen.moveTo((x0, y0)); pen.lineTo((x1, y0)); pen.lineTo((x1, y1)); pen.lineTo((x0, y1)); pen.closePath()
    return p

def union(paths):
    out = Path()
    for p in paths:
        out = op(out, p, PathOp.UNION)
    return out

def stacked(weight):
    """(PS5, LM, cursor) paths. The cursor takes the 5's width and the cap height of the second line."""
    top = glyphs('PS5', weight, 0, CAP)
    base2 = CAP + LEAD * CAP + CAP
    bot = glyphs('LM', weight, 0, base2)
    five = top[2].bounds
    return union(top), union(bot), rect(five[0], base2 - CAP, five[2], base2)

def horizontal(weight):
    g = glyphs('PS5LM', weight, 0, CAP)
    # The cursor takes the sixth mono cell, drawn like the 5's ink in its own cell, as in the stacked logo.
    cell = 600 + TRACK
    five = g[2].bounds
    x0 = five[0] + 3 * cell
    return union(g[:3]), union(g[3:]), rect(x0, 0, x0 + five[2] - five[0], CAP)

def write(path, parts, w, h, s, ox, oy, bg=None, title='PS5LM'):
    body = [f'  <rect width="{w:g}" height="{h:g}" fill="{bg}"/>'] if bg else []
    for colour, p in parts:
        pen = SVGPathPen(None, ntos=lambda v: f'{v:.2f}'.rstrip('0').rstrip('.'))
        p.draw(TransformPen(pen, (s, 0, 0, s, ox, oy)))
        body.append(f'  <path fill="{colour}" d="{pen.getCommands()}"/>')
    open(os.path.join(OUT, path), 'w').write(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}">\n  <title>{title}</title>\n' + '\n'.join(body) + '\n</svg>\n')

def bounds(paths):
    return union(paths).bounds

def logo(path, shape, ink, weight, height=256):
    """Transparent logo, trimmed to its ink plus nothing: clear space is the user's job (see README)."""
    a, b, c = shape(weight)
    x0, y0, x1, y1 = bounds([a, b, c])
    s = height / (y1 - y0)
    write(path, [(ink, a), (ACCENT, b), (ink, c)], round((x1 - x0) * s, 2), height, s, -x0 * s, -y0 * s)

def tile(path, weight, size=512, fill=0.72, lift=0.012):
    """The app icon: the stacked logo on the ground colour, full bleed (the PS5 and iOS round the corners)."""
    a, b, c = stacked(weight)
    x0, y0, x1, y1 = bounds([a, b, c])
    s = size * fill / max(x1 - x0, y1 - y0)
    ox = (size - (x1 - x0) * s) / 2 - x0 * s
    oy = (size - (y1 - y0) * s) / 2 - y0 * s - size * lift     # optical centre, a touch above the middle
    write(path, [(INK_DARK, a), (ACCENT, b), (INK_DARK, c)], size, size, s, ox, oy, bg=GROUND)

def banner(path, w, h, tagline='llama.cpp on PlayStation 5'):
    """Social image and PS5 home screen background: the site's grid and periwinkle glow, the horizontal logo low
    on the right with the tagline under it. Everything scales with the height."""
    g, cap = h / 18, 0.136 * h
    x0, base = 0.5625 * w, 0.76 * h
    parts = []
    a, b, c = horizontal(760)
    s = cap / CAP
    lx = bounds([a, b, c])[0]
    for colour, p in ((INK_DARK, a), (ACCENT, b), (INK_DARK, c)):
        parts.append((colour, '', p, s, x0 - lx * s, base - CAP * s))
    t = union(glyphs(tagline, 400, 0, CAP, track=0))
    ts = 0.035 * h / 1000
    parts.append(('#ebebf5', ' fill-opacity="0.55"', t, ts, x0 - t.bounds[0] * ts, base + 0.074 * h - CAP * ts))
    background(path, w, h, parts)

def background(path, w, h, parts):
    g = h / 18
    body = []
    for colour, extra, p, sc, ox, oy in parts:
        pen = SVGPathPen(None, ntos=lambda v: f'{v:.2f}'.rstrip('0').rstrip('.'))
        p.draw(TransformPen(pen, (sc, 0, 0, sc, ox, oy)))
        body.append(f'  <path fill="{colour}"{extra} d="{pen.getCommands()}"/>')
    open(os.path.join(OUT, path), 'w').write(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">
  <title>PS5LM</title>
  <defs>
    <radialGradient id="glow" cx="78%" cy="70%" r="60%">
      <stop offset="0" stop-color="{ACCENT}" stop-opacity="0.22"/>
      <stop offset="0.55" stop-color="{ACCENT}" stop-opacity="0.04"/>
      <stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/>
    </radialGradient>
    <pattern id="grid" width="{g:g}" height="{g:g}" patternUnits="userSpaceOnUse">
      <path d="M{g:g} 0H0V{g:g}" fill="none" stroke="#ebebf5" stroke-opacity="0.035" stroke-width="{h / 1080:g}"/>
    </pattern>
    <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{GROUND}" stop-opacity="1"/>
      <stop offset="0.45" stop-color="{GROUND}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="{GROUND}"/>
  <rect width="{w}" height="{h}" fill="url(#grid)"/>
  <rect width="{w}" height="{h}" fill="url(#fade)"/>
  <rect width="{w}" height="{h}" fill="url(#glow)"/>
''' + '\n'.join(body) + '\n</svg>\n')

def card(path, w, h, lines=(('Run LLMs and decision models', INK_DARK), ('on your PS5', ACCENT)),
         sub="llama.cpp on the console's GPU, for every device on your network"):
    """Link previews: the logo top left, a two-line headline as on the site's hero (the second line in the
    accent), one muted line under it, over the same grid and glow as the banner. Sizes follow the height."""
    k, x = h / 630, 80 * h / 630
    items = []
    def put(paths, colour, scale, left, base, extra=''):
        u = union(paths) if isinstance(paths, list) else paths
        items.append((colour, extra, u, scale, left - u.bounds[0] * scale, base - CAP * scale))
    a, b, c = horizontal(760)
    ls, lx = 40 * k / CAP, x - bounds([a, b, c])[0] * 40 * k / CAP
    for colour, p in ((INK_DARK, a), (ACCENT, b), (INK_DARK, c)):
        items.append((colour, '', p, ls, lx, 84 * k))
    em, base = 64 * k, 334 * k
    for text, colour in lines:
        put(glyphs(text, 760, 0, CAP, track=-45), colour, em / 1000, x, base)
        base += 1.2 * em
    put(glyphs(sub, 400, 0, CAP, track=0), '#ebebf5', 25 * k / 1000, x, base - 1.2 * em + 72 * k, ' fill-opacity="0.6"')
    background(path, w, h, items)

# On dark grounds the light letters glow and look heavier, so the reversed versions are set a step lighter.
logo('ps5lm-stacked-dark.svg', stacked, INK_DARK, 760)
logo('ps5lm-stacked-light.svg', stacked, INK_LIGHT, 800)
logo('ps5lm-horizontal-dark.svg', horizontal, INK_DARK, 760)
logo('ps5lm-horizontal-light.svg', horizontal, INK_LIGHT, 800)
tile('ps5lm-icon.svg', 760)
# Small cut for 16-48 px: heavier letters, less margin, so the strokes survive the pixel grid.
tile('ps5lm-icon-small.svg', 900, fill=0.84, lift=0)
# Wide formats: the link preview (og:image), GitHub's social preview, the PS5 home screen background (pic0/pic1).
card('ps5lm-og.svg', 1200, 630)
card('ps5lm-social.svg', 1280, 640)
banner('ps5lm-background.svg', 3840, 2160)

# Self-check: the cursor sits exactly under the 5, and the two lines keep their gap.
a, b, c = stacked(800)
five = glyphs('PS5', 800, 0, CAP)[2].bounds
assert abs(c.bounds[0] - five[0]) < 1e-6 and abs(c.bounds[2] - five[2]) < 1e-6
assert abs((c.bounds[1] - CAP) - LEAD * CAP) < 1e-6
print('ok', sorted(os.listdir(OUT)))
