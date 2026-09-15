import numpy as np


# ============================================================
# EASY MAZES
# ============================================================

EASY_MAZE = [
    list("S......"),
    list(".###..."),
    list("...#..."),
    list(".###.#."),
    list(".....#."),
    list(".#####."),
    list("......G")
]


EASY_MEADOW = [
    list("S......"),
    list("..##..."),
    list("......."),
    list(".##...."),
    list("......."),
    list("...##.."),
    list("......G")
]


EASY_BREEZE = [
    list("S......"),
    list(".##...."),
    list("......."),
    list("....##."),
    list("......."),
    list(".##...."),
    list("......G")
]


EASY_FLOWER_PATH = [
    list("S.#...."),
    list("..#...."),
    list("......."),
    list("....#.."),
    list("....#.."),
    list("......."),
    list("......G")
]


EASY_LITTLE_GROVE = [
    list("S......"),
    list(".#....."),
    list(".#..#.."),
    list("....#.."),
    list("..#...."),
    list("......."),
    list("......G")
]


# ============================================================
# MEDIUM MAZES
# ============================================================

MEDIUM_MAZE = [
    list("S.#...."),
    list("..#.#.."),
    list("..#.#.."),
    list("..#.#.."),
    list("....#.."),
    list(".#####."),
    list("......G")
]


MEDIUM_WINDING_GARDEN = [
    list("S..#..."),
    list("##.#..."),
    list("...#.#."),
    list(".#...#."),
    list(".###..."),
    list(".....##"),
    list("......G")
]


MEDIUM_TWISTED_PATH = [
    list("S...#.."),
    list(".##.#.."),
    list("...##.."),
    list("..#...."),
    list("..#.##."),
    list("......#"),
    list("......G")
]


MEDIUM_HEDGE_MAZE = [
    list("S.#...."),
    list("..#..#."),
    list("..##.#."),
    list(".....#."),
    list(".###..."),
    list("...##.."),
    list("......G")
]


MEDIUM_FOREST_TRAIL = [
    list("S...#.."),
    list(".##.#.."),
    list("...#..."),
    list(".###.#."),
    list(".....#."),
    list("..##..."),
    list("......G")
]


# ============================================================
# HARD MAZES
# ============================================================

HARD_MAZE = [
    list("S..#..."),
    list("##.#.#."),
    list("...#.#."),
    list(".###.#."),
    list(".....#."),
    list(".#####."),
    list("......G")
]


HARD_TANGLED_GROVE = [
    list("S.#...."),
    list("..#.#.."),
    list("#.#.#.."),
    list("...##.."),
    list(".#...#."),
    list(".###..."),
    list("......G")
]


HARD_DEEP_FOREST = [
    list("S...#.."),
    list("###.#.."),
    list("..#.#.."),
    list(".#....."),
    list(".#.###."),
    list("...#..."),
    list("......G")
]


HARD_CHAOS_GARDEN = [
    list("S.#.#.."),
    list("....#.."),
    list("##..#.."),
    list("...###."),
    list(".#....."),
    list(".#####."),
    list("......G")
]


HARD_MOUNTAIN_TRAIL = [
    list("S..##.."),
    list("##...#."),
    list("...##.."),
    list(".#..#.."),
    list(".#..###"),
    list(".....#."),
    list("......G")
]


# ============================================================
# MAZE INFORMATION
# ============================================================

MAZE_INFO = {

    # ---------------- EASY ----------------

    "Easy Garden": (
        "🌱",
        "A gentle first garden",
        "Few walls • great for understanding the basics"
    ),

    "Easy Meadow": (
        "🌼",
        "A peaceful meadow",
        "Simple turns • a friendly introduction"
    ),

    "Easy Breeze": (
        "🍃",
        "The breezy garden",
        "Light obstacles • easy exploration"
    ),

    "Easy Flower Path": (
        "🌸",
        "The flower trail",
        "A few barriers • follow the open garden"
    ),

    "Easy Little Grove": (
        "🌿",
        "A tiny grove",
        "Small detours • gentle Q-learning challenge"
    ),

    # ---------------- MEDIUM ----------------

    "Medium Garden": (
        "🌿",
        "A winding little world",
        "More barriers • stronger exploration required"
    ),

    "Medium Winding Garden": (
        "🌳",
        "The winding garden",
        "Several turns • the agent must explore carefully"
    ),

    "Medium Twisted Path": (
        "🌀",
        "The twisted trail",
        "Tighter routes • more exploration required"
    ),

    "Medium Hedge Maze": (
        "🌱",
        "The hedge maze",
        "Dense barriers • careful decisions matter"
    ),

    "Medium Forest Trail": (
        "🌲",
        "The forest trail",
        "Multiple turns • a stronger learning challenge"
    ),

    # ---------------- HARD ----------------

    "Hard Garden": (
        "🌲",
        "The tangled grove",
        "Tricky turns • a tougher learning challenge"
    ),

    "Hard Tangled Grove": (
        "🌳",
        "The tangled grove",
        "Dense barriers • exploration becomes important"
    ),

    "Hard Deep Forest": (
        "🌲",
        "The deep forest",
        "Narrow routes • difficult exploration"
    ),

    "Hard Chaos Garden": (
        "🍂",
        "The chaotic garden",
        "Complex barriers • a serious Q-learning challenge"
    ),

    "Hard Mountain Trail": (
        "⛰️",
        "The mountain trail",
        "Tight passages • the hardest garden challenge"
    ),
}


# ============================================================
# GET MAZE
# ============================================================

def get_maze(name):

    mazes = {
        "Easy Garden": EASY_MAZE,
        "Easy Meadow": EASY_MEADOW,
        "Easy Breeze": EASY_BREEZE,
        "Easy Flower Path": EASY_FLOWER_PATH,
        "Easy Little Grove": EASY_LITTLE_GROVE,

        "Medium Garden": MEDIUM_MAZE,
        "Medium Winding Garden": MEDIUM_WINDING_GARDEN,
        "Medium Twisted Path": MEDIUM_TWISTED_PATH,
        "Medium Hedge Maze": MEDIUM_HEDGE_MAZE,
        "Medium Forest Trail": MEDIUM_FOREST_TRAIL,

        "Hard Garden": HARD_MAZE,
        "Hard Tangled Grove": HARD_TANGLED_GROVE,
        "Hard Deep Forest": HARD_DEEP_FOREST,
        "Hard Chaos Garden": HARD_CHAOS_GARDEN,
        "Hard Mountain Trail": HARD_MOUNTAIN_TRAIL,
    }

    if name in mazes:
        return np.array(mazes[name], dtype=str)

    raise ValueError("Unknown maze selected.")