# Kolko i krzyzyk - uczenie ze wzmocnieniem
# Pierwsze kilka funkcji sluzy do generowania wszystkich mozliwych
# ukladow wystepujacych w grze (z dokladnoscia do obrotow i symetrii)
# i do wypisywania ich w konsoli. Przyjmujemy, ze zawsze kolko zaczyna.
# Uklady to lista 10-elementowa, gdzie jej n-ty element to lista
# mozliwych do uzyskania plansz zawierajacych dokladnie n symboli
# tzn. zerowy element listy ukladow to jednoelemetowa lista, ktorej
# jedynym elementem jest pusta plansza; pierwszy element list to
# 3-elementowa lista, ktorej elementami sa: plansza z kolkiem w rogu,
# plansza z kolkiem na srodku krawedzi i plansza z kolkiem na samym
# srodku (przypominamy o symetriach i obrotach - bez tego plansz
# 1-elementowych byloby 9). Plansza z kolei to macierz 3x3, gdzie
# 0 oznacza pole puste, 1 - kolko i -1 - krzyzyk.
import numpy as np
import pickle
from os.path import exists


# wypisuje symbol zamiast wart. liczbowej
def printsign(val):
    if val == -1:
        print('X', end='')
    elif val == 0:
        print(' ', end='')
    elif val == 1:
        print('O', end='')


# wypisuje wiersz z symbolami
def printrow(row):
    printsign(row[0])
    print('|', end='')
    printsign(row[1])
    print('|', end='')
    printsign(row[2])
    print()


# wypisuje searator wierszy
def printline():
    print("-+-+-")


# wypisuje cala plansze
def print_board(board):
    printrow(board[0])
    printline()
    printrow(board[1])
    printline()
    printrow(board[2])


# sprawdza czy dwie plansze sa izomorficzne (tzn. czy mozna
# nalozyc jedna na druga obrotami lub symetriami)
def boardsdiff(board1, board2):
    tmp = board2.copy()
    if np.array_equal(board1, tmp):
        return False
    tmp = np.rot90(tmp)
    if np.array_equal(board1, tmp):
        return False
    tmp = np.rot90(tmp)
    if np.array_equal(board1, tmp):
        return False
    tmp = np.rot90(tmp)
    if np.array_equal(board1, tmp):
        return False
    tmp = np.rot90(tmp)
    tmp = np.flipud(tmp)
    if np.array_equal(board1, tmp):
        return False
    tmp = np.rot90(tmp)
    if np.array_equal(board1, tmp):
        return False
    tmp = np.rot90(tmp)
    if np.array_equal(board1, tmp):
        return False
    tmp = np.rot90(tmp)
    if np.array_equal(board1, tmp):
        return False
    return True


# sprawdza, czy na planszy jest uklad wygrywajacy
# zwraca 1 jesli wygraly kolka, -1 jesli krzyzyki i 0 wpp
def winner(board):
    for k in range(0, 3):
        if board[k][0] == board[k][1] == board[k][2] == 1:
            return 1
        elif board[k][0] == board[k][1] == board[k][2] == -1:
            return -1
        elif board[0][k] == board[1][k] == board[2][k] == 1:
            return 1
        elif board[0][k] == board[1][k] == board[2][k] == -1:
            return -1
    if board[0][0] == board[1][1] == board[2][2] == 1:
        return 1
    elif board[0][0] == board[1][1] == board[2][2] == -1:
        return -1
    elif board[0][2] == board[1][1] == board[2][0] == 1:
        return 1
    elif board[0][2] == board[1][1] == board[2][0] == -1:
        return -1
    return 0


# generuje liste wszystkich plansz zawierajacych n elementow
# w tym celu korzysta z listy plansz zawierajacyhc n-1 elementow
# (list_of_boards)
def next_layer(list_of_boards, n):
    layer = []
    val = 2 * (n % 2) - 1  # wartosc dodawanej liczby - jesli n - nieparz. to 1, -1 wpp
    for board in list_of_boards:
        for x in range(0, 3):
            for y in range(0, 3):  # petla po wwszystkich planszach mniejszego rozmiaru i po wszytkich polach
                board_clone = board.copy()
                if board[x][y] == 0 and winner(board) == 0:
                    board_clone[x][y] = val  # wykonujemy ruch w wolnym miejscu na sensownej planszy
                    if not layer:
                        layer.append(board_clone)
                    else:
                        diff = True
                        for cor in layer:  # sprawdzamy czy wczesniej nie uzyskalismy takiej planszy
                            if not boardsdiff(cor, board_clone):
                                diff = False
                        if diff:  # dodajemy dobra plansze do listy
                            layer.append(board_clone)
    return layer


# generuje liste list wszystkich mozliwych do uzyskania plansz
def generate_boards():
    board = np.zeros(shape=(3, 3))
    boards = list()
    boards.append([board])
    for i in range(1, 10):
        boards.append(next_layer(boards[i-1], i))
    return boards

# lista wszystkich mozliwych ukladow w grze
file_name = "generated_data/boards_save"

if not exists(file_name):
    BOARDS = generate_boards()
    with open(file_name, 'wb') as f:
        pickle.dump(BOARDS, f)

