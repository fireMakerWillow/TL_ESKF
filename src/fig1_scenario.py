"""Figure 1 — cooperative localization and centralized covariance storage.

All mobile platforms are called entities. The central computer may be
a dedicated device or be hosted by one of the entities. The covariance
callout is schematic: it does not introduce another ranging edge.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent

ANCHOR_C = "#1B7F5F"
INTER_C = "#D9820B"
ENTITY_C = "#0077BB"
INK = "#242424"
NOTE = "#505861"
BLUE = "#3B5EA6"

ANCHORS = {
    "n1": (8, 65),
    "n2": (92, 67),
    "n3": (49, 7),
}

# More evenly distributed mobile entities
ENTITIES = {
    "g1": ((27, 54), 25),
    "g2": ((73, 54), 155),
    "g3": ((50, 38), -40),
    "g4": ((23, 25), 70),
    "g5": ((81, 27), 120),
}

ANCHOR_LINKS = [
    ("g1", "n1"),
    ("g1", "n2"),
    ("g2", "n2"),
    ("g3", "n3"),
]

INTER_LINKS = [
    ("g1", "g3"),
    ("g2", "g3"),
    ("g1", "g4"),
    ("g3", "g4"),
    ("g3", "g5"),
    ("g2", "g5"),
]

ANCHOR_FREE = "g4"


def anchor_icon(ax, x, y, s=1, z=6):
    h, w = 13*s, 5*s

    ax.add_patch(
        Polygon(
            [(x-w, y), (x-.34*w, y+h), (x+.34*w, y+h), (x+w, y)],
            closed=False,
            fill=False,
            ec=INK,
            lw=1.4,
            zorder=z
        )
    )

    for f in (.24, .56, .86):
        half = w * (1 - .66*f)
        ax.plot(
            [x-half, x+half],
            [y+h*f]*2,
            color=INK,
            lw=.8,
            zorder=z
        )

    ax.plot(
        [x-w, x+.34*w],
        [y, y+h],
        color=INK,
        lw=.7,
        zorder=z
    )

    ax.plot(
        [x+w, x-.34*w],
        [y, y+h],
        color=INK,
        lw=.7,
        zorder=z
    )

    top = y + h + 1.6*s

    ax.add_patch(
        Circle(
            (x, top),
            1.5*s,
            fc=INK,
            ec="none",
            zorder=z+1
        )
    )

    for r, lw in ((4.8*s, 1.4), (7.8*s, 1.0)):
        for start in (26, 120):
            ax.add_patch(
                Arc(
                    (x, top),
                    2*r,
                    2*r,
                    theta1=start,
                    theta2=start+34,
                    ec=INK,
                    lw=lw,
                    zorder=z
                )
            )


def entity_icon(ax, x, y, heading=0, s=1, z=6):
    r = 6*s

    ax.add_patch(
        Circle(
            (x, y),
            r,
            fc="#CFE3F2",
            ec=INK,
            lw=1.4,
            zorder=z
        )
    )

    a = np.radians(heading)

    forward = np.array([
        np.cos(a),
        np.sin(a)
    ])

    perp = np.array([
        -np.sin(a),
        np.cos(a)
    ])

    c = np.array([x, y])

    triangle = [
        c + .88*r*forward,
        c - .42*r*forward + .52*r*perp,
        c - .42*r*forward - .52*r*perp
    ]

    ax.add_patch(
        Polygon(
            triangle,
            closed=True,
            fc=ENTITY_C,
            ec=INK,
            lw=.9,
            joinstyle="round",
            zorder=z+1
        )
    )


def uwb_module(ax, x, y, s=1, z=9):
    ax.add_patch(
        Circle(
            (x, y),
            2.5*s,
            fc="#3A3A3A",
            ec=INK,
            lw=.8,
            zorder=z
        )
    )

    ax.add_patch(
        Circle(
            (x-.7*s, y+.7*s),
            .8*s,
            fc="#8C8C8C",
            ec="none",
            zorder=z+1
        )
    )


def imu_module(ax, x, y, s=1, z=9):
    ax.add_patch(
        Rectangle(
            (x-2.6*s, y-1.9*s),
            5.2*s,
            3.8*s,
            fc="#E8E4DA",
            ec=INK,
            lw=.9,
            zorder=z
        )
    )

    for k in range(3):
        yy = y - 1.1*s + k*1.1*s

        ax.plot(
            [x-3.6*s, x-2.6*s],
            [yy]*2,
            color=INK,
            lw=.7,
            zorder=z
        )

        ax.plot(
            [x+2.6*s, x+3.6*s],
            [yy]*2,
            color=INK,
            lw=.7,
            zorder=z
        )

    ax.add_patch(
        Rectangle(
            (x-1.2*s, y-.8*s),
            2.4*s,
            1.6*s,
            fc="#B9B2A4",
            ec="none",
            zorder=z+1
        )
    )


def radio_module(ax, x, y, s=1, z=9):
    for dx in (-1.8*s, 1.8*s):
        ax.plot(
            [x+dx, x+dx*1.5],
            [y+1.3*s, y+4.2*s],
            color=INK,
            lw=1,
            zorder=z
        )

    ax.add_patch(
        FancyBboxPatch(
            (x-3.4*s, y-1.6*s),
            6.8*s,
            2.9*s,
            boxstyle="round,pad=0,rounding_size=0.7",
            fc="#3A3A3A",
            ec=INK,
            lw=.9,
            zorder=z+1
        )
    )

    for k in range(3):
        ax.add_patch(
            Circle(
                (x-1.8*s+k*1.8*s, y-.2*s),
                .45*s,
                fc="#9BD3C4",
                ec="none",
                zorder=z+2
            )
        )


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Liberation Sans",
        "DejaVu Sans"
    ],
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(7.15, 5.05))


# ---------------------------------------------------------------------
# Range edges
# ---------------------------------------------------------------------

# Anchor-to-entity ranges
for g, n in ANCHOR_LINKS:
    (gx, gy), (nx, ny) = ENTITIES[g][0], ANCHORS[n]

    ax.annotate(
        "",
        xy=(gx, gy),
        xytext=(nx, ny+13),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.20,head_length=0.46",
            color=ANCHOR_C,
            lw=1.35,
            alpha=.94,
            mutation_scale=16,
            shrinkA=7,
            shrinkB=15
        ),
        zorder=2
    )


# Inter-entity ranges
for u, v in INTER_LINKS:
    (ux, uy), (vx, vy) = ENTITIES[u][0], ENTITIES[v][0]

    ax.annotate(
        "",
        xy=(vx, vy),
        xytext=(ux, uy),
        arrowprops=dict(
            arrowstyle="<|-|>,head_width=0.19,head_length=0.42",
            color=INTER_C,
            lw=1.35,
            ls=(0, (3.2, 2.2)),
            mutation_scale=16,
            shrinkA=15,
            shrinkB=15
        ),
        zorder=3
    )


# ---------------------------------------------------------------------
# Anchors and entities
# ---------------------------------------------------------------------

for nx, ny in ANCHORS.values():
    anchor_icon(ax, nx, ny)

    ax.text(
        nx,
        ny-3,
        "UWB anchor",
        ha="center",
        va="top",
        fontsize=8.1,
        color=INK,
        zorder=7
    )


for (gx, gy), heading in ENTITIES.values():
    entity_icon(
        ax,
        gx,
        gy,
        heading
    )


# ---------------------------------------------------------------------
# Anchorless-entity explanation
# ---------------------------------------------------------------------

fx, fy = ENTITIES[ANCHOR_FREE][0]

ax.annotate(
    "No direct anchor range:\n"
    "absolute information arrives\n"
    "via the cooperative graph",
    xy=(fx-4.5, fy-4.2),
    xytext=(fx-1, fy-13),
    ha="center",
    va="top",
    fontsize=7.6,
    color=NOTE,
    linespacing=1.3,
    arrowprops=dict(
        arrowstyle="-",
        color=NOTE,
        lw=.8,
        ls=(0, (2.2, 2.2)),
        shrinkA=2,
        shrinkB=3
    ),
    zorder=7
)


# ---------------------------------------------------------------------
# Devices carried by each entity
# ---------------------------------------------------------------------

bx, by, bw, bh = 27, 79, 46, 14

ax.add_patch(
    FancyBboxPatch(
        (bx, by),
        bw,
        bh,
        boxstyle="round,pad=0,rounding_size=1.6",
        fc="white",
        ec=BLUE,
        lw=1.2,
        zorder=8
    )
)

for k, (draw, label) in enumerate((
    (uwb_module, "UWB"),
    (imu_module, "IMU"),
    (radio_module, "Radio")
)):
    cx = bx + bw/3*(k+.5)

    draw(
        ax,
        cx,
        by+bh-5.4
    )

    ax.text(
        cx,
        by+1.8,
        label,
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=INK,
        zorder=10
    )


ax.text(
    bx+bw/2,
    by+bh+1.4,
    "Carried by every entity",
    ha="center",
    va="bottom",
    fontsize=8.3,
    fontweight="bold",
    color=BLUE
)

ax.annotate(
    "",
    xy=(27.5, 64.0),
    xytext=(34, 86),
    arrowprops=dict(
        arrowstyle="-",
        color=BLUE,
        lw=.9,
        ls=(0, (2.2, 2.2)),
        shrinkA=1,
        shrinkB=1
    ),
    zorder=7
)


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------

entity_legend = (
    Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=9,
        markerfacecolor="#CFE3F2",
        markeredgecolor=INK,
        markeredgewidth=1
    ),
    Line2D(
        [],
        [],
        linestyle="none",
        marker=">",
        markersize=4.8,
        markerfacecolor=ENTITY_C,
        markeredgecolor=INK,
        markeredgewidth=.7
    ),
)

ax.legend(
    handles=[
        Line2D(
            [],
            [],
            color=ANCHOR_C,
            lw=1.6
        ),
        Line2D(
            [],
            [],
            color=INTER_C,
            lw=1.6,
            ls=(0, (3.2, 2.2))
        ),
        entity_legend
    ],
    labels=[
        "Anchor-to-entity range",
        "Inter-entity range",
        "Mobile entity"
    ],
    handler_map={
        tuple: HandlerTuple(
            ndivide=1,
            pad=0
        )
    },
    loc="upper center",
    bbox_to_anchor=(.5, -.008),
    ncol=3,
    frameon=False,
    fontsize=8.0,
    handlelength=2.1,
    columnspacing=1.35,
    handletextpad=.6
)


ax.set_xlim(-4, 106)
ax.set_ylim(-2, 98)
ax.set_aspect("equal")
ax.axis("off")

fig.subplots_adjust(
    left=.025,
    right=.975,
    top=.99,
    bottom=.065
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

for suffix, kwargs in (
    (".pdf", {}),
    (".png", {"dpi": 320})
):
    fig.savefig(
        OUT / f"fig1_system_scenario{suffix}",
        bbox_inches="tight",
        pad_inches=.10,
        **kwargs
    )

plt.close(fig)

print("Saved Figure 1 PDF and PNG")