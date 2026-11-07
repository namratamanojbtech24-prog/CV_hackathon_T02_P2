"""Shared visual identity (dark canvas, one hot accent). Colours validated with the dataviz palette tools:
categorical slots 1-4 of the reference palette stepped for a dark surface."""
SURFACE = "#0B0B0C"
PANEL = "#151517"
LINE = "#2A2A2D"
INK = "#F2F0EB"
INK2 = "#B9B7AF"
MUTED = "#7C7B76"
ACCENT = "#FF5A1F"

# one colour per entity, never re-assigned
COLORS = {
    "Traditional ISP": "#3987E5",   # blue
    "Hybrid ISP": "#FF5A1F",        # accent orange: the hero
    "Learned ISP": "#199E70",       # aqua
    "Hybrid-Lite": "#C98500",       # yellow (distilled)
    "Hybrid INT8": "#9085E9",       # violet (quantised)
    "Unprocessed (naive)": "#7C7B76",
    "Ground truth": "#F2F0EB",
}
DASH = {"Traditional ISP": "solid", "Hybrid ISP": "solid", "Learned ISP": "dash", "Hybrid-Lite": "dot",
        "Hybrid INT8": "dashdot", "Unprocessed (naive)": "solid"}
MPL_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}
