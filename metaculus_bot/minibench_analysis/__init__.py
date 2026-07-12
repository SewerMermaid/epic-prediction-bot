"""MiniBench results analysis.

Two entry points (see ``cli.py``):

- ``two-sessions-ago``: top-10 bots + my bot for the MiniBench two sessions back
  (two back because the most recent one may still have unresolved questions that
  close later).
- ``all-except-current``: my bot only, across every resolved MiniBench.

Accuracy uses a two-tier scheme (see ``scoring.py`` for the exact rules):

- Tier 1 ("beat chance"), applied to *every* bot and every question type: a
  forecast counts as accurate iff it beat a maximally ignorant baseline
  (>50% on binary, >1/N on the resolved MC option, above-uniform density on the
  resolved numeric bin). This is the sign of the Metaculus baseline score, is
  un-gameable, and is uniform across types — so it is the primary metric and the
  only one computable for other bots from forecast data alone.
- Tier 2 (intuitive breakdown), applied to *my* bot only (where we have full
  forecast data): binary directional, MC arg-max, and numeric resolved-within
  P25-P75. Peer score is reported alongside where the API exposes it.
"""
