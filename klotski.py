import numpy as np
from dataclasses import dataclass


@dataclass
class Block:
    """
    An individual sliding block, with position and shape properties
    """

    id: int  # unique ID for each block
    r: int  # row of top-left corner
    c: int  # column of top-left corner
    h: int  # height (number of rows)
    w: int  # width (number of columns)

    def __post_init__(self):
        # Ensure row is 0-4 and column is 0-3 (4x5 board)
        if (not isinstance(self.r, int)) or (not isinstance(self.c, int)):
            raise TypeError(
                f"Row and column number must be integers; got r={self.r!r} and c={self.c!r}"
            )
        if (self.r < 0) or (self.r > 4) or (self.c < 0) or (self.c > 3):
            raise ValueError(
                f"Row number must be 0-4 and column number must be 0-3; got r={self.r!r} and c={self.c!r}"
            )

        # Ensure height and width are 1-2
        if (not isinstance(self.h, int)) or (not isinstance(self.w, int)):
            raise TypeError(
                f"Height and width must be integers; got h={self.h!r} and w={self.w!r}"
            )
        if (self.h < 1) or (self.h > 2) or (self.w < 1) or (self.h > 2):
            raise ValueError(
                f"Height and width must be 1-2; got h={self.h!r} and w={self.w!r}"
            )


class MainBlock(Block):
    """
    The main block which must be moved to the winning position; size is fixed to 2x2
    """

    def __init__(self, id: int, r: int, c: int):
        super().__init__(id=id, r=r, c=c, h=2, w=2)

    def __post_init__(self):
        super().__post_init__()
        if (self.h != 2) or (self.w != 2):
            raise ValueError(
                "Height and width of the main block must be 2; this should not be set by the user"
            )


class Board:
    """
    The board that holds blocks
    """

    def __init__(self):
        self._initialise_board()

    def _add_block(self, r: int, c: int, h: int, w: int):
        """
        Creates a block object and records its occupancy on the board
        """
        self.check_valid_position(r=r, c=c, h=h, w=w)

        idx = max(self.blocks.keys()) + 1
        self.blocks[idx] = Block(idx, r=r, c=c, h=h, w=w)
        self.occupancy[r : r + h, c : c + w] = True

    def _add_main_block(self, r: int, c: int):
        """
        Creates a block object (main 2x2 block) and records its occupancy on the board
        """
        self.check_valid_position(r=r, c=c, h=2, w=2)

        idx = 0 if len(self.blocks) == 0 else max(self.blocks.keys()) + 1
        self.blocks[idx] = MainBlock(idx, r=r, c=c)
        self.occupancy[r : r + 2, c : c + 2] = True

    def check_valid_position(self, r: int, c: int, h: int, w: int):
        """
        Raises an error if a block is being added or moved to a location that is already occupied
        """
        if self.occupancy[r : r + h, c : c + w].any():
            raise ValueError(
                f"Cannot add/move a block to this position because it is already occupied: r={r!r}, c={c!r}, h={h!r}, w={w!r}; occupancy:\n{self.occupancy!r}"
            )

    def _initialise_board(self):
        """
        Adds blocks to the board in their starting positions
        """
        self.occupancy = np.full((5, 4), False)
        self.blocks = dict()
        self._add_main_block(r=0, c=1)
        self._add_block(r=0, c=0, h=1, w=1)
        self._add_block(r=0, c=3, h=1, w=1)
        self._add_block(r=1, c=0, h=1, w=1)
        self._add_block(r=1, c=3, h=1, w=1)
        self._add_block(r=2, c=0, h=2, w=1)
        self._add_block(r=2, c=1, h=1, w=1)
        self._add_block(r=2, c=2, h=1, w=1)
        self._add_block(r=2, c=3, h=2, w=1)
        self._add_block(r=3, c=1, h=1, w=1)
        self._add_block(r=3, c=2, h=1, w=1)
        self._add_block(r=4, c=0, h=1, w=1)
        self._add_block(r=4, c=3, h=1, w=1)

        # TODO: move this to a test file
        n_unoccupied = self.occupancy.size - self.occupancy.sum()
        assert n_unoccupied == 2, (
            f"Board has been incorrectly initialised; should be 2 unoccupied spaces but there are {n_unoccupied}. Occupancy: {self.occupancy}"
        )
