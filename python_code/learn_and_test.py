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
import pickle
import numpy as np
from random import randint
from python_code import boards
from python_code import common_funct as cf
from python_code import qlearning as ql

from python_code import sarsa as sa

# wczytanie uzyskanych wczesniej wynikow
with open("generated_data/boards_save", 'rb') as f1:
    BOARDS = pickle.load(f1)
with open("generated_data/big_q-learning_table", 'rb') as f2:
    BIG_QLEARNING_TABLE = pickle.load(f2)
with open("generated_data/big_sarsa_table", 'rb') as f3:
    BIG_SARSA_TABLE = pickle.load(f3)

# lista akcji U - up, M - mid, D - down, L - left, R - right
ACTIONS = ['UL', 'UM', 'UR', 'ML', 'MM', 'MR', 'DL', 'DM', 'DR']

# parametry
EPSILON = 0.95  # strategia zachlanna
ALPHA = 0.3  # wspolczynnik wartosci kroku
GAMMA = 0.8  # wspolczynnik dyskontowania
MAX_EPISODES = 10000  # ilosc gier
TEST_EPISODE = 250  # czestosc porownywania
TEST_GAMES = 100  # ilosc gier testowych
REWARDS = [0, 1]  # nagrody


# funkcja wybierajaca ruch przy stosowaniu strategii z luka
# zwraca numer pozycji po wykonaniu wybranego ruchu
def move_with_gap(mv_num, pos_num, qtable, gap):
    possible_actions = cf.give_possible_actions(mv_num, pos_num)
    act_val = -10000  # arbitralna bardzo mala wartosc rownowazna z -nieskonczonosc
    for act_id in possible_actions:  # wyszukanie najwiekszej wartosci dla mozliwych akcji
        if qtable[mv_num - 1][pos_num][act_id] > act_val:
            act_val = qtable[mv_num - 1][pos_num][act_id]
    act_in_gap = list()
    for act_id in possible_actions:  # dodanie wszystkich akcji mieszczacych sie w luce
        if qtable[mv_num - 1][pos_num][act_id] >= act_val - gap:
            act_in_gap.append(act_id)
    act = np.random.choice(act_in_gap)  # wybor losowej z zatwierdzonych akcji
    return cf.position_after_move(mv_num, pos_num, act)


# symuluje komplet wszystkich 6 mozliwych gier dla graczy o tabelach funkcji
# Q: qtable1 i qtable2 (wszystkie 6 mozliwych to kombinacje gracza rozpoczynajacego
#  i pierwszego ruchu); zwraca ilosc wygranych, przegranych i remisow gracza 1
def deterministic_game(qtable1, qtable2):
    wins = 0
    loses = 0
    ties = 0
    for game_num in range(6):
        starting_player = game_num % 2
        first_mv = game_num % 3
        mv_num = 2
        pos_num = first_mv
        end = False
        while (not end) and (mv_num <= 9):
            if mv_num % 2 == starting_player:
                pos_num = cf.move_against_player(mv_num, pos_num, qtable2)
                if boards.winner(BOARDS[mv_num][pos_num]) != 0:
                    end = True
                    loses += 1
            else:
                pos_num = cf.move_against_player(mv_num, pos_num, qtable1)
                if boards.winner(BOARDS[mv_num][pos_num]) != 0:
                    end = True
                    wins += 1
            mv_num += 1
        if not end:
            ties += 1
    return ties, loses, wins


# rozgrywa pojedynacza gre dla graczy o tabelach qtable1, qtable2
# obaj stosuja strategie zachlanna z luka; zwracany jest wynik gry:
# 1 jesli gracz1 wygral, -1 jesli orzegral, 0 jesli byl remis
def nondeterministc_game(qtable1, qtable2):
    gap1 = qtable1[0][0][:].max() - qtable1[0][0][:].min()
    gap2 = qtable2[0][0][:].max() - qtable2[0][0][:].min()
    starting_player = randint(0, 1)  # losowanie pierwszego gracza
    mv_num = 1
    pos_num = 0
    while mv_num <= 9:  # petla symulujaca gre
        if mv_num % 2 == starting_player:  # ruch pierwszego gracza
            pos_num = move_with_gap(mv_num, pos_num, qtable2, gap2)
            if boards.winner(BOARDS[mv_num][pos_num]) != 0:
                return -1
        else:  # ruch drugiego gracza
            pos_num = move_with_gap(mv_num, pos_num, qtable1, gap1)
            if boards.winner(BOARDS[mv_num][pos_num]) != 0:
                return 1
        mv_num += 1
    return 0


# rozgrywa komplet 6 gier dla danego gracza z kazdym graczy wzorcowych,
# wypisuje rezultat do pliku
def deterministic_games(qtable, f):
    ties_q, loses_q, wins_q = deterministic_game(qtable, BIG_QLEARNING_TABLE)
    ties_s, loses_s, wins_s = deterministic_game(qtable, BIG_SARSA_TABLE)
    print("Ties:", ties_q + ties_s, "         Q-learning -", ties_q, "SARSA -", ties_s, file=f)
    print("Loses:", loses_q + loses_s, "        Q-learning -", loses_q, "SARSA -", loses_s, file=f)
    print("Wins:", wins_q + wins_s, "         Q-learning -", wins_q, "SARSA -", wins_s, file=f)


# rozgrywa TEST_GAMES gier kazda z losowym graczem wzorcowym
# zlicza wyniki i na koniec wypisuje do pliku
def nondeterministc_games(qtable, f):
    wins_q = 0
    wins_s = 0
    loses_q = 0
    loses_s = 0
    ties_q = 0
    ties_s = 0
    for game in range(TEST_GAMES):
        opp = randint(0, 1)  # losowanie przeciwnika
        if opp == 0:  # wylosowano Q-learning
            res = nondeterministc_game(qtable, BIG_QLEARNING_TABLE)
            wins_q += max(res, 0)
            loses_q -= min(res, 0)
            ties_q += (res + 1) % 2
        else:  # wylosowano SARSA
            res = nondeterministc_game(qtable, BIG_SARSA_TABLE)
            wins_s += max(res, 0)
            loses_s -= min(res, 0)
            ties_s += (res + 1) % 2
    print("Ties:", ties_q + ties_s, "         Q-learning -", ties_q, "SARSA -", ties_s, file=f)
    print("Loses:", loses_q + loses_s, "        Q-learning -", loses_q, "SARSA -", loses_s, file=f)
    print("Wins:", wins_q + wins_s, "         Q-learning -", wins_q, "SARSA -", wins_s, file=f)


# rozgrywa TEST_GAMES gier miedzy graczami poslugujacymi sie tabelami
# qtable1 i qtable2; gracze stosuja strategie zachlanna z luka;
# wypisuje wynik gier do pliku
def nondeterministc_versus_games(qtable1, qtable2, f):
    wins = 0
    loses = 0
    ties = 0
    for game in range(TEST_GAMES):
        res = nondeterministc_game(qtable1, qtable2)
        wins += max(res, 0)
        loses -= min(res, 0)
        ties += (res + 1) % 2
    print("Ties:", ties, file=f)
    print("Q-learning loses:", loses, file=f)
    print("Q-learning wins:", wins, file=f)


# zarzadza kolejnoscia rozgrywania gier walidayjnych oraz wypisuje
# do pliku separatory miedzy grami
def play_games(episode, qtable1, qtable2, f):
    print("--------------------", file=f)
    print("Episode:", episode, file=f)
    print("Deterministic for Q-learning:", file=f)
    deterministic_games(qtable1, f)
    print("Algorythm gap:", qtable1[0][0][:].max() - qtable1[0][0][:].min(), file=f)
    print("Nondeterministic for Q-learning:", file=f)
    nondeterministc_games(qtable1, f)
    print(".....................", file=f)
    print("Deterministic for SARSA:", file=f)
    deterministic_games(qtable2, f)
    print("Algorythm gap:", qtable2[0][0][:].max() - qtable2[0][0][:].min(), file=f)
    print("Nondeterministic for SARSA:", file=f)
    nondeterministc_games(qtable2, f)
    print(".....................", file=f)
    print("Deterministic versus games:", file=f)
    ties, wins_s, wins_q = deterministic_game(qtable1, qtable2)
    print("Ties:", ties, "   Q-learning wins:", wins_q, "    SARSA wins:", wins_s, file=f)
    print("Nondeterministic versus games", file=f)
    nondeterministc_versus_games(qtable1, qtable2, f)


# glowna czesc programu - przeprowadza serie procesow uczenia dla roznych
# zestawow parametrow, w czasie kazdego procesu uczy jednego ucznia stosujac Q-learning
# i jednego SARSA, kazdy uczy sie na podstawie MAX_EPISODE gier i co TEST_EPISODE gier
# jest poddawany walidacji; wyniki zapisuje do wskazanego pliku
with open("results/results2_1.txt", "w") as f:
    # glowna czesc programu
    while EPSILON <= 0.95:  # petle po roznych parametrach EPSILON, ALPHA i GAMMA
        ALPHA = 0.3
        while ALPHA <= 0.61:
            GAMMA = 0.8
            while GAMMA <= 0.96:
                # wypisywanie separatorow i informacji o parametrach do pliku
                print("========================", file=f)
                print("========================", file=f)
                print("========================", file=f)
                print("EPSILON", EPSILON, sep=' ', end='\n', file=f)
                print("ALPHA", ALPHA, sep=' ', end='\n', file=f)
                print("GAMMA", GAMMA, file=f)
                print("Q-learning gap:", BIG_QLEARNING_TABLE[0][0][:].max() - BIG_QLEARNING_TABLE[0][0][:].min(),
                      file=f)
                print("SARSA gap:", BIG_SARSA_TABLE[0][0][:].max() - BIG_SARSA_TABLE[0][0][:].min(), file=f)
                # zerowanie tabel uczniow
                QTABLE_QLEARNING = ql.generate_qtable()
                QTABLE_SARSA = sa.generate_qtable()
                # petla po licznie walidacji
                for period in range(MAX_EPISODES // TEST_EPISODE):
                    # nauka Q-learningiem przed kolejna walidacja
                    for episode in range(TEST_EPISODE):
                        prev_action = ACTIONS[0]  # incjacja zmiennych do nauki
                        mv_num = 1
                        prev_pos = 0
                        curr_pos = 0
                        end = False
                        while (not end) and (mv_num <= 9):  # petla dla pojedynczej rozgrywki
                            action = cf.choose_action(mv_num, curr_pos, EPSILON, QTABLE_QLEARNING)
                            curr_pos, prev_pos, prev_action, end = ql.make_action(prev_action, action, mv_num,
                                                                                  curr_pos, prev_pos, QTABLE_QLEARNING)
                            mv_num += 1
                    # nauka SARSA przed kolejna walidacja
                    for episode in range(TEST_EPISODE):
                        prev_action = ACTIONS[0]  # incjacja zmiennych do nauki
                        mv_num = 1
                        prev_pos = 0
                        curr_pos = 0
                        action = cf.choose_action(mv_num, curr_pos, EPSILON, QTABLE_SARSA)
                        end = False
                        while (not end) and (mv_num <= 9):  # petla dla pojedynczej rozgrywki
                            curr_pos, prev_pos, prev_action, action, end = sa.make_action(prev_action, action,
                                                                                          mv_num,curr_pos,
                                                                                          prev_pos, QTABLE_SARSA)
                            mv_num += 1
                    play_games((period + 1) * TEST_EPISODE, QTABLE_QLEARNING, QTABLE_SARSA, f)  # walidacja
                GAMMA += 0.05
            ALPHA += 0.05
        EPSILON += 0.05
