"""Run the exhaustive visual-projection brain screen once (results cached in spikes/screen/out/screen_cache.jsonl)."""
from flylab import screen
rows = screen.brain_screen("visual_projection", ["MDN", "GF"], progress=True)
print("done", len(rows))
