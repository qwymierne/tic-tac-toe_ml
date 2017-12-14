# Funkcje potrzebne zarowno Q-learningowi jak i SARSA
# podaczas rozwiazywania kolka i krzyzyka, zarowno przy
# nauce jak i grze z graczem
import pickle

import numpy as np

from python_code import boards

# lista akcji U - up, M - mid, D - down, L - left, R - right
ACTIONS = ['UL', 'UM', 'UR', 'ML', 'MM', 'MR', 'DL', 'DM', 'DR']

# wczytujemy uklad wszystkich plansz
with open("generated_data/boards_save", 'rb') as f:
    BOARDS = pickle.load(f)


# dla danej planszy zwraca liste ruchow, ktore mozna na niej wykonac
def give_possible_actions(mv_num, position_num):
    if mv_num == 10:  # wykorzystane tylko w SARSA na potrzeby ostatniego ruchu
        return [0]
    board = BOARDS[mv_num - 1][position_num]
    pos_actions = []
    for x in range(0, 3):
        for y in range(0, 3):
            if board[x][y] == 0:
                pos_actions.append(3 * x + y)
    return pos_actions


# losuje ruch z najlepszych mozliwych ruchow tzn. takich, ktore mozna
# wykonac i maja najwieksza wartosc funkcji Q (w szczegolnosci,
# jesli tylko jeden ruch ma najwieksza wartosc to on zostanie
# wybrany, zas jesli wszystkie maja rowna wartosc, to wybierze losowy)
def best_action(possible_act, qrow):
    maxi = -10000  # mala arbitralna wartosc rownowazna -nieskonczonosc
    eq_best_actions = list()
    for i in possible_act:
        if qrow[i] > maxi:
            eq_best_actions.clear()
            eq_best_actions.append(i)
            maxi = qrow[i]
        elif qrow[i] == maxi:
            eq_best_actions.append(i)
    return np.random.choice(eq_best_actions)


# wybiera ruch, ktory zostanie wykonany - sprawdza jakie ruchy sa dostepne
# nastepnie losuje jednostajnie z odcinka [0,1] - jesli wylosuje liczbe
# >= EPSILON to losuje dowolny mozliwy ruch, wpp wybiera ruch korzystjac
# z funkcji best_action
def choose_action(mv_num, pos_num, epsilon, qtable):
    pos_actions = give_possible_actions(mv_num, pos_num)
    if np.random.uniform() > epsilon:
        act_name = ACTIONS[np.random.choice(pos_actions)]
    else:
        act_name = ACTIONS[best_action(pos_actions, qtable[mv_num - 1][pos_num])]
    return act_name


# zwraca faktyczne maksimum następnego ruchu, tzn. maksymalna wartosc w tabeli
# wsrod mozliwych akcji
def true_max(mv_num, pos_num, qtable):
    maxi = 0
    posibble_act = give_possible_actions(mv_num + 1, pos_num)
    for act in posibble_act:
        if qtable[mv_num][pos_num][act] > maxi:
            maxi = qtable[mv_num][pos_num][act]
    return maxi


# wybiera ruch zgodnie ze strategia zachlanna (bez szansy na ruch losowy)
# jedynie pierwszy ruch jest losowany
# zwraca numer pozycji po ruchu
def move_against_player(mv_num, pos_num, qtable):
    if mv_num == 1:
        act = np.random.choice([0, 1, 4])
    else:
        possible_actions = give_possible_actions(mv_num, pos_num)
        act_val = -10000  # bardzo mala wartosc rownowaznie -nieskonczonosc
        for act_id in possible_actions:
            if qtable[mv_num - 1][pos_num][act_id] > act_val:
                act = act_id
                act_val = qtable[mv_num - 1][pos_num][act_id]
    return position_after_move(mv_num, pos_num, act)


# wykonuje ruch i zwraca numer pozycji po jego wykoaniu
def position_after_move(mv_num, pos_num, act_num):
    val = 2 * (mv_num % 2) - 1  # jesli gra kolko val = 1, krzyzyk - val = -1
    x = act_num // 3
    y = act_num % 3  # wspolrzedne wykonanego ruchu

    curr_board = BOARDS[mv_num - 1][pos_num].copy()  # robocza kopia planszy
    curr_board[x][y] = val  # wykonanie ruchu

    # wyszukanie pozycji, ktora odpowiada planszy po ruchu
    new_pos = 0
    while boards.boardsdiff(BOARDS[mv_num][new_pos], curr_board):
        new_pos += 1
    return new_pos


# pyta gracza o to jaki ruch chce wykonac, wykonuje go i zwraca numer nowej pozycji
def player_move(mv_num, pos_num):
    possible_actions = give_possible_actions(mv_num, pos_num)
    print("Possible actions:", possible_actions)
    act = int(input("Choose action: "))
    return position_after_move(mv_num, pos_num, act)
