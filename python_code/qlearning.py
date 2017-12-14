# Funcje związane z algorytmem Q-learning w kolku i krzyku;
# glowna czesc programu oferuje uzytkownikowi gre
# z nauczonym programem programem
import pickle
import numpy as np

from os.path import exists

from python_code import boards
from python_code import common_funct as cf

# lista akcji U - up, M - mid, D - down, L - left, R - right
ACTIONS = ['UL', 'UM', 'UR', 'ML', 'MM', 'MR', 'DL', 'DM', 'DR']


# parametry programu
EPSILON = 0.9  # strategia zachlanna
ALPHA = 0.5  # wspolczynnik wartosci kroku
GAMMA = 0.9  # wspolczynnik dyskontowania
MAX_EPISODES = 20000  # ilosc gier
REWARDS = [0, 1]  # nagrody

# wczytujemy uklad wszystkich plansz
file_obj = open("generated_data/boards_save", 'rb')
BOARDS = pickle.load(file_obj)

# Zaznaczmy w tym miejscu, ze w zadnym momencie nie "tworzymy"
# planszy gry, ktora sie toczy, poruszamy sie jedynie miedzy planszami
# z wygenerowanej listy wszystkich mozliwych plansz tzn. pamietamy, ktory
# numer planszy z tej listy mamy danym momencie (i, dodatkowo, jaki byl w
# poprzednim ruchu, bo jest nam potrzebny). Czyli innymi slowy sytuacje na planszy
# rozpoznajemy korzystajac z: listy wszystkich plansz, ilosci symboli na planszy,
# numeru planszy na liscie plansz o danej liczby symboli


# genetruje "tabele" wartosci funkcji Q
# jest to 10-el. lista tabel o wymiarach:
# (ilosc plansz w n-tej warstwie)x(ilosc moziwych ruchow = 9)
# warto zauwazyc, ze qtabele o numerach parzystych "naleza" do kolka
# o nieparzystych - do krzyzyka (bo kolko zawsze zaczyna, czyli w trakcie
# gry bedzie startowac od 0, 2, 4... symboli na planszy, krzyzyk odwrotnie)
def generate_qtable():
    table = []
    for layer in BOARDS:
        new_table = np.zeros((len(layer), len(ACTIONS)))
        table.append(new_table)
    return table


QTABLE = generate_qtable()


# uaktualnia tabele wartosci funkcji Q zarowno dla kolka jak i krzyzyka
# tzn. jesli ostatni ruch wykonalo kolko to przynaje uaktualnia zarowno
# tabele dla tego ruchu kolka jak i poprzedniego ruchu krzyzyka (dlatego
# musimy pamietac sytuacje rowniez jedna kolejke wczesniej), jesli jako
# ostatni ruszal sie krzyzyk - analogicznie
def q_update(prev_action, action, mv_num, new_pos, curr_pos, prev_pos, reward, qtable):

    curr_act = ACTIONS.index(action)
    prev_act = ACTIONS.index(prev_action)

    # stare warosci w tabeli
    q_curr_mv_val = qtable[mv_num - 1][curr_pos][curr_act]
    q_prev_mv_val = qtable[mv_num - 2][prev_pos][prev_act]

    # wartosc optymalnego ruchu w nowym stanie
    q_opt_val = cf.true_max(mv_num, new_pos, qtable)

    # wartosc uaktualnienia w tabeli zgodna z Q-learningiem
    q_change_curr = reward + (-1) * GAMMA * q_opt_val
    # ponizej nagroda *(-1), bo krzyzyk ma miec wartosci o przeciwnym znaku do kolka
    q_change_prev = (-1) * reward + GAMMA * q_opt_val

    # uaktualnienie tabel kolka i krzyzyka
    qtable[mv_num - 1][curr_pos][curr_act] += ALPHA * (q_change_curr - q_curr_mv_val)
    qtable[mv_num - 2][prev_pos][prev_act] += ALPHA * (q_change_prev - q_prev_mv_val)


# wykonuje zaplanowana akcje przy okazji uaktualniajac przy tabele wartosci funkcji Q
# zwraca (kolejno) numer pozycji po wykoaniu akcji, numer pozycji przed wykoaniem akcji,
# nazwe wykoananej akcji, infromacje czy osiagnieto pozycje wygrywajaca
def make_action(prev_action, action, mv_num, curr_pos, prev_pos, qtable):
    new_pos = cf.position_after_move(mv_num, curr_pos, ACTIONS.index(action))
    # uaktualnianie qtabeli w zaleznosci od tego czy gra sie skonczyla
    if boards.winner(BOARDS[mv_num][new_pos]) != 0:
        q_update(prev_action, action, mv_num, new_pos, curr_pos, prev_pos, REWARDS[1], qtable)
        end = True
    else:
        q_update(prev_action, action, mv_num, new_pos, curr_pos, prev_pos, REWARDS[0], qtable)
        end = False
    return new_pos, curr_pos, action, end

# plik, w k torym ma sie znajdowac wygenerowana tabela
file_name = "generated_data/big_q-learning_table"


# sprawdzenie czy tabela wartosci funkcji Q istnieje -
# jesli nie to jest tworzona na podstawie MAX_EPISODES
# gier z podaynmi wyzej parametrami
if not exists(file_name):
    for episode in range(MAX_EPISODES):
        prev_action = ACTIONS[0]  # incjacja zmiennych
        mv_num = 1
        prev_pos = 0
        curr_pos = 0
        end = False
        while (not end) and (mv_num <= 9):  # petla dla pojedynczej rozgrywki
            action = cf.choose_action(mv_num, curr_pos, EPSILON, QTABLE)
            curr_pos, prev_pos, prev_action, end = make_action(prev_action, action, mv_num, curr_pos, prev_pos, QTABLE)
            mv_num += 1
        with open(file_name, 'wb') as f:
            pickle.dump(QTABLE, f)
else:
    with open(file_name, 'rb') as f:
        QTABLE = pickle.load(f)


# glowna czesc programu - gra z uzytkownikiem
print("Play as O: 1; Play as X: 2; End: 0")
new_game = int(input("Choose your symbol: "))
while new_game != 0:
    mv_num = 1
    pos_num = 0
    end = False  # incjacja zmiennych
    while (not end) and (mv_num <= 9):  # petla dla pojedynczej rozgrywki
        active_player = (mv_num + new_game) % 2
        # wybieranie i wykonanie akcji
        if active_player == 0:
            pos_num = cf.player_move(mv_num, pos_num)
        else:
            pos_num = cf.move_against_player(mv_num, pos_num, QTABLE)
        if boards.winner(BOARDS[mv_num][pos_num]) != 0:
            end = True
        boards.print_board(BOARDS[mv_num][pos_num])
        print("-----------------")  # wypisanie planszy po ruchu
        mv_num += 1
    print("Play as O: 1; Play as X: 2; End: 0")
    new_game = int(input("Choose your symbol: "))
