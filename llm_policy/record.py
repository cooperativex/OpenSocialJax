"""Record GIFs of the LLM playing: one file per trial, a caption strip on
every frame (trial, step, tool, the skill being executed and the model's
reasoning), frames every `every` env steps."""
from __future__ import annotations

import os

import numpy as onp
from PIL import Image, ImageDraw, ImageFont


class GifRecorder:
    def __init__(self, env, out_dir, tag, trials=(0, 1, 2), every=2, fps=12, craft=False, scale=0.6, max_steps=None):
        self.env, self.out_dir, self.tag, self.scale, self.max_steps = env, out_dir, tag, scale, max_steps
        self.trials, self.every, self.fps, self.craft = set(trials), every, fps, craft
        self.frames, self.current = [], None
        self.caption = ""
        os.makedirs(out_dir, exist_ok=True)
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 13)
        except Exception:
            self.font = ImageFont.load_default()
        self.written = []

    def set_caption(self, text):
        self.caption = text

    def __call__(self, state, rec, trial_ended):
        trial = rec["trial"]
        if trial not in self.trials:
            if trial_ended:
                self.flush(trial)
            return
        if self.current is None:
            self.current = trial
        # regular cadence, plus every step the agent fires and the step after
        # (the beam is drawn for a single step, and the cleared cell shows next)
        fired_agents = rec.get("fired_agents", [0] if rec.get("fired") else [])
        fired_now = bool(fired_agents)
        take = (rec["t"] % self.every == 0) or fired_now or self._fired_prev
        self._fired_prev = fired_now
        if take and (self.max_steps is None or rec["t"] < self.max_steps):
            img = Image.fromarray(onp.asarray(self.env.render(state)).astype(onp.uint8))
            if fired_now:
                img = self._draw_beam(img, state, fired_agents)
            if self.scale != 1.0:
                img = img.resize((int(img.width * self.scale), int(img.height * self.scale)), Image.NEAREST)
            w, h = img.size
            strip = 34
            canvas = Image.new("RGB", (w, h + strip), (24, 24, 24))
            canvas.paste(img, (0, strip))
            d = ImageDraw.Draw(canvas)
            tool = f"tool {rec['held']}" if not self.craft else ("TOOL OBTAINED" if rec.get("crafted") else "no tool")
            line1 = f"trial {trial + 1}  step {rec['t']:4d}  {tool}  reward so far {self._reward:.0f}"
            d.text((6, 3), line1, fill=(240, 240, 240), font=self.font)
            d.text((6, 18), self.caption[:110], fill=(180, 200, 255), font=self.font)
            self.frames.append(canvas)
        self._reward += rec["reward"]
        if trial_ended:
            self.flush(trial)

    _reward = 0.0
    _fired_prev = False

    _STEP = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def _draw_beam(self, img, state, agents=(0,)):
        """The environment only marks the beam on empty ground, so a shot at
        the river is invisible in the render: overlay the beam footprint
        (front 1, front 2, front-left, front-right — the env's own targets)
        as a translucent yellow, on the firing frame only."""
        R, C = self.env.GRID_SIZE_ROW, self.env.GRID_SIZE_COL
        tile = img.width / (C + 2)                 # the render keeps a one-tile border
        cells = []
        for a in agents:
            loc = onp.asarray(state.agent_locs)[a]
            r, c, d = int(loc[0]), int(loc[1]), int(loc[2])
            f = self._STEP[d]; rt = self._STEP[(d + 1) % 4]; lt = self._STEP[(d - 1) % 4]
            cells += [(r + f[0], c + f[1]), (r + 2 * f[0], c + 2 * f[1]),
                      (r + f[0] + rt[0], c + f[1] + rt[1]), (r + f[0] + lt[0], c + f[1] + lt[1])]
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d2 = ImageDraw.Draw(over)
        for j, (rr, cc) in enumerate(cells):
            i = j % 4
            if not (0 <= rr < R and 0 <= cc < C):
                continue
            x0, y0 = (cc + 1) * tile, (rr + 1) * tile
            pad = 3 if i < 2 else 8
            d2.rectangle([x0 + pad, y0 + pad, x0 + tile - pad, y0 + tile - pad],
                         fill=(255, 220, 40, 150 if i < 2 else 100), outline=(255, 245, 120, 230), width=2)
        return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")

    def flush(self, trial):
        if self.frames:
            path = os.path.join(self.out_dir, f"{self.tag}_trial{trial + 1}.gif")
            self.frames[0].save(path, save_all=True, append_images=self.frames[1:],
                                duration=int(1000 / self.fps), loop=0, optimize=False)
            self.written.append(path)
            print(f"[gif] wrote {path} ({len(self.frames)} frames)", flush=True)
        self.frames, self.current, self._reward = [], None, 0.0
