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
        """
        Checks inputs are valid for row, column, height and width
        """
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

    def _check_moves(self, occupancy):
        """
        Checks whether the block can move north, south, east, or west
        """
        self.can_move_north = True
        self.can_move_east = True
        self.can_move_south = True
        self.can_move_west = True

        # Remove possibility of moving beyond the edge of the board, or occupying an already-occupied space
        if self.r == 0:  # Top row: cannot move north
            self.can_move_north = False
        elif (
            occupancy[self.r - 1, self.c : self.c + self.w] >= 0
        ).any():  # Another block occupying at least one of the spaces above: cannot move north
            self.can_move_north = False
        if self.r == 4:  # Bottom row: cannot move south
            self.can_move_south = False
        elif (
            occupancy[self.r + 1, self.c : self.c + self.w] >= 0
        ).any():  # Another block occupying at least one of the spaces below: cannnot move south
            self.can_move_south = False
        if self.c == 0:  # Left column: cannot move west
            self.can_move_west = False
        elif (
            occupancy[self.r : self.r + self.h, self.c - 1] >= 0
        ).any():  # Another block occupying at least one of the spaces to the left: cannot move west
            self.can_move_west = False
        if self.c == 3:  # Right column: cannot move east
            self.can_move_east = False
        elif (
            occupancy[self.r : self.r + self.h, self.c + 1] >= 0
        ).any():  # Another block occuping at least one of the spaces to the right: cannot move east
            self.can_move_east = False

        self.n_moves = sum(
            [
                self.can_move_north,
                self.can_move_east,
                self.can_move_south,
                self.can_move_west,
            ]
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

        idx = 0 if len(self.blocks) == 0 else max(self.blocks.keys()) + 1
        self.blocks[idx] = Block(idx, r=r, c=c, h=h, w=w)
        self.occupancy[r : r + h, c : c + w] = idx

    def _add_main_block(self, r: int, c: int):
        """
        Creates a block object (main 2x2 block) and records its occupancy on the board
        """
        self.check_valid_position(r=r, c=c, h=2, w=2)

        idx = 0 if len(self.blocks) == 0 else max(self.blocks.keys()) + 1
        self.blocks[idx] = MainBlock(idx, r=r, c=c)
        self.occupancy[r : r + 2, c : c + 2] = idx

    def check_valid_position(self, r: int, c: int, h: int, w: int):
        """
        Raises an error if a block is being added or moved to a location that is already occupied
        """
        if (self.occupancy[r : r + h, c : c + w] >= 0).any():
            raise ValueError(
                f"Cannot add/move a block to this position because it is already occupied: r={r!r}, c={c!r}, h={h!r}, w={w!r}; occupancy:\n{self.occupancy!r}"
            )

    def _initialise_board(self):
        """
        Adds blocks to the board in their starting positions. Includes an occupancy map where unoccupied locations are -1 and occupied ones take the index of the occupying block
        """
        self.occupancy = np.full((5, 4), -1)
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
        n_unoccupied = np.sum(self.occupancy == -1)
        assert n_unoccupied == 2, (
            f"Board has been incorrectly initialised; should be 2 unoccupied spaces but there are {n_unoccupied}. Occupancy:\n{self.occupancy}"
        )

    def find_valid_moves(self) -> list:
        """
        Examines the occupancy map to identify all possible moves on the board
        """
        valid_moves = []
        for block_id, block in self.blocks.items():
            block._check_moves(self.occupancy)
            if block.can_move_north:
                valid_moves.append((block_id, "north"))
            if block.can_move_east:
                valid_moves.append((block_id, "east"))
            if block.can_move_south:
                valid_moves.append((block_id, "south"))
            if block.can_move_west:
                valid_moves.append((block_id, "west"))

        # TODO: move this to a test file
        assert len(valid_moves) > 0, (
            f"No valid moves found; occupancy:\{self.occupancy}"
        )

        # TODO: replace print statement with logs and visualisation
        print(f"Found {len(valid_moves)} block(s) that can move:\n{valid_moves}")

        return valid_moves

    def move_block(self, block_id: int, direction: str):
        """
        Updates the row or column of a block, and the occupancy map
        """

        # Check that the block to be moved corresponds to one that exists
        if block_id not in self.blocks:
            raise ValueError(
                f"Block ID {block_id} does not exist; possible IDs are {self.blocks.keys()!r}"
            )

        # Calculate new row or column, moving one space to the north/east/south/west
        old_r = self.blocks[block_id].r
        old_c = self.blocks[block_id].c
        if direction == "north":
            new_r = old_r - 1
            new_c = old_c
        elif direction == "south":
            new_r = old_r + 1
            new_c = old_c
        elif direction == "east":
            new_r = old_r
            new_c = old_c + 1
        elif direction == "west":
            new_r = old_r
            new_c = old_c - 1
        else:
            raise ValueError(
                f'Direction must be "north", "east", "south", or "west"; got {direction!r}'
            )

        # Update the block's row and column, replace its old occupancy with -1, and replace its new occupancy with the block ID
        self.check_valid_position(
            r=new_r, c=new_c, h=self.blocks[block_id].h, w=self.blocks[block_id].w
        )
        self.blocks[block_id].r = new_r
        self.blocks[block_id].c = new_c
        self.occupancy[old_r, old_c] = -1
        self.occupancy[new_r, new_c] = block_id

    def take_random_move(self):
        """
        Identifies the valid moves, selects one at random, and updates the board
        """
        old_occupancy = self.occupancy.copy()
        valid_moves = self.find_valid_moves()
        move_idx = np.random.randint(0, len(valid_moves))
        block_id, direction = valid_moves[move_idx]
        self.move_block(block_id, direction)

        # TODO: move these to a test file
        assert (self.occupancy != old_occupancy).any(), (
            f"Occupancy has not been correctly updated:\n{self.occupancy}"
        )
        n_unoccupied = np.sum(self.occupancy == -1)
        assert n_unoccupied == 2, (
            f"Board has been incorrectly initialised; should be 2 unoccupied spaces but there are {n_unoccupied}. Occupancy:\n{self.occupancy}"
        )
