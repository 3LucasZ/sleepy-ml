import chess

# Unicode symbols for chess pieces
pieces = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    None: '·'  # Empty square
}

def print_chessboard(board):
    print('  a b c d e f g h')  # Column labels
    print(' +----------------+')
    for rank in range(7, -1, -1):  # Iterate from rank 8 to 1
        row = f'{rank + 1} |'
        for file in range(8):  # Iterate over files a to h
            square = chess.square(file, rank)  # Get square index
            piece = board.piece_at(square)  # Get piece at square
            row += f'{pieces[piece.symbol() if piece else None]} '
        row += f'| {rank + 1}'
        print(row)
    print(' +----------------+')
    print('  a b c d e f g h')

# Create a new chess board
board = chess.Board()

# Print the initial chessboard
print_chessboard(board)