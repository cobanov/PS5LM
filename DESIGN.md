# PS5LM design

How PS5LM looks on the web: the site (`site/index.html`, live at ps5lm.cobanov.dev) and any page built next
to it, such as API docs or a tool page. The logo, its files and its rules are in [`brand/`](brand/README.md).
The site is the reference implementation: when this file and the site disagree, the site is newer.

The app on the TV (`ps5/app/dashboard.cpp`) has its own graphite UI in Manrope, made to be read from a sofa.
Web pages use the language below.

## Character

A dark, quiet page with one periwinkle accent, bold mono headings and plain sentences. The richness lives in
one place: the top of the page, which carries the console's home screen background (a faint grid and a
periwinkle glow) behind the headline. Everything else is flat: hairlines, one surface step, no shadows, no
gradients, no glow.

## Tokens

Copy this block as is. Every colour on a page comes from it.

```css
@font-face { font-family: "Recursive Sans"; src: url(/assets/fonts/recursive-sans-v1.woff2) format("woff2"); font-weight: 300 1000; font-display: swap; }
@font-face { font-family: "Recursive Mono"; src: url(/assets/fonts/recursive-mono-v1.woff2) format("woff2"); font-weight: 300 1000; font-display: swap; }
:root {
  color-scheme: dark;
  --paper: #0e0e14;                      /* the ground */
  --panel: #16161d;                      /* one step up: panels, code, buttons, the footer */
  --ink: #f2f2f7;                        /* text */
  --muted: rgba(235, 235, 245, 0.6);     /* secondary text, nav links */
  --rule: rgba(235, 235, 245, 0.1);      /* the one hairline */
  --soft: #22222c;                       /* number badges */
  --accent: #8288f0;                     /* periwinkle: the primary button, links, LM in the logo */
  --accent-ink: #0e0e14;                 /* text on the accent: white on it is only 3.1:1 */
  --hl-key: #c4b5fd;                     /* code: flags, keys */
  --hl-str: #86efac;                     /* code: strings */
  --hl-fn: #a5aaf6;                      /* code: commands */
  --hl-num: #fdba74;                     /* code: numbers */
  --radius: 6px;                         /* panels, frames, images */
  --radius-sm: 4px;                      /* buttons, small controls */
  --sans: "Recursive Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "Recursive Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
```

The fonts are self-hosted from `site/assets/fonts` (latin, variable weight 300–1000), under the SIL Open Font
License in `recursive-OFL.txt`. Preload both.

## Type

- **Mono, bold** for h1, h2, buttons, labels, table heads and figures. **Sans** for everything you read.
- h1: `clamp(2.125rem, 5.5vw, 3.25rem)`, weight 700, line height 1.08, letter spacing -0.045em, `text-wrap: balance`.
  A second line in the accent is the page's one coloured phrase.
- h2: `clamp(1.5rem, 3vw, 1.875rem)`, weight 700, letter spacing -0.035em, then a `.sub` line in muted 1.0625rem.
- Body 16px/1.5. Lede 1.1875rem muted, at most 40ch. Notes and captions 0.8125rem muted.
- Labels 1rem mono 700, letter spacing -0.03em. Small text never gets negative letter spacing.
- `font-variant-numeric: tabular-nums` on the body.

## Layout

- One column, `max-width: 1040px`, 20px gutters. The hero text is at most 680px wide.
- Header: the horizontal logo at 30px tall on the left (`brand/ps5lm-horizontal-dark.svg`), muted nav links at
  0.9375rem on the right. Under 480px, links marked `.wide` hide.
- Sections are 72px apart. Inside a section: h2, the sub line, then the content.
- Grids auto-fit: `minmax(min(100%, 280px), 1fr)` for short items, `420px` for longer ones and code panels.
- Everything stacks at phone width with no sideways scroll of the page. A wide table scrolls inside its frame.

## The top of the page

```css
.top {
  background:
    linear-gradient(180deg, transparent 50%, var(--paper)),
    radial-gradient(55% 75% at 80% 35%, rgba(130, 136, 240, 0.2), rgba(130, 136, 240, 0.04) 55%, transparent),
    linear-gradient(90deg, var(--paper) 0, transparent 45%),
    repeating-linear-gradient(0deg, rgba(235, 235, 245, 0.04) 0 1px, transparent 1px 60px),
    repeating-linear-gradient(90deg, rgba(235, 235, 245, 0.04) 0 1px, transparent 1px 60px),
    var(--paper);
}
```

The grid fades in from the left, the glow sits top right, and the band fades into the ground at the bottom. Use it
once per page, behind the header and the hero.

## Components

Take the markup and CSS from `site/index.html`:

- **Buttons** (`.btn`): mono 500, `12px 20px`, `--radius-sm`, a hairline on the panel colour. One primary per view
  (`.btn.primary`: accent ground, `--accent-ink` text, hover `#969bf3`) and at most one secondary next to it.
- **Points** (`.points`): a hairline on top, a bold mono label, one muted line under it. `.points.two` for longer
  answers. `.num` puts a numbered badge before a label.
- **Code panels** (`.code`): a head with a number badge, a label, the file or endpoint in muted mono and a Copy button.
  The code is 0.875rem/1.65 mono, highlighted by hand: `.c` comment, `.k` flag or key, `.s` string, `.f` command,
  `.n` number. Each line is a `.ln` span with a hanging indent. Break lines by hand at 44 characters at most, so they
  fit a phone at 0.75rem. Every command has to run as written.
- **Tables** (`.compare`): a frame on the panel colour, a sticky first column, muted mono heads, figures in `.fig`.
  Marks and figures, not sentences. Use `.q` for quiet marks like "Required".
- **Images and video**: a hairline border and `--radius`, at full column width or two to a row with a caption whose first
  word is bold.
- **CTA** (`.cta`): a centred panel with h2, one sub line and the primary button.
- **Footer**: the panel colour, an about column with the icon (`site/media/icon.png`), three link columns with
  muted heads, then a legal line: "Made by Mert Cobanov", the version, and the line saying PS5LM isn't affiliated
  with Sony.
- **Focus**: `outline: 2px solid var(--accent); outline-offset: 3px` on links, buttons and video.

## Words

- English, sentence case, plain words. Headings, labels and buttons end without a period.
- Say what it does with numbers people can check: "Qwen 3.5 9B at 55 tokens a second", not "blazing fast".
- No exclamation marks, no emoji, no hype words.
- Name Sony's products only to say what PS5LM runs on. Never use PlayStation's logo, button symbols or console
  shapes.

## Head and icons

- `<link rel="icon" href="/media/favicon.png">` plus `/favicon.ico` at the root, and
  `<link rel="apple-touch-icon" href="/media/icon.png">`. The sources are in `brand/web/`.
- `og:image` is `brand/ps5lm-og.svg` rendered at 1200×630, with `twitter:card` set to `summary_large_image`.
- `<meta name="theme-color" content="#0e0e14">`.
