import chess

board = chess.Board()

print(board)
print(board.legal_moves)

board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
board.push_san("Bc4")
board.push_san("Nf6")
board.push_san("Qxf7")

print(board.is_checkmate())
print(board)

