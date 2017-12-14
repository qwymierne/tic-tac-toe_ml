# Program do czytania uzyskanych rezultatow i na ich podstawie generujacy
# serie wykresow: gry przegrane przy zmodyfikowanej strategii zachlannej,
# gry przegrane przy strategii zachlannej z luka, luka, roziear luki
# a procent przegranych gier

import numpy as np
import matplotlib.pyplot as plt
import os

POINT_NUMBER = 40  # liczba walidacji = liczba punktow na wykresie

# tabele zbierajace kolejne liczby do jako wspolrzedne
episodes = np.zeros(shape=POINT_NUMBER)
q_det_loses = np.zeros(shape=POINT_NUMBER)
s_det_loses = np.zeros(shape=POINT_NUMBER)
q_ndet_loses = np.zeros(shape=POINT_NUMBER)
s_ndet_loses = np.zeros(shape=POINT_NUMBER)
q_alg_gap = np.zeros(shape=POINT_NUMBER)
s_alg_gap = np.zeros(shape=POINT_NUMBER)
# parametry wypisywane do wykresow
epsilon = 0
alpha = 0
gamma = 0
eq_count = 0
# zmienne kontrolujace obecnie czytany fragment
qswitch_det = False
sswitch_det = False
qswitch_ndet = False
sswitch_ndet = False
det_versus_switch = False
ndet_versus_switch = False


# rysuje i zapisuje wykres dla Q-learningu i SARSA ilosci przegranych przy grze strategia
# zachlanna w czasie, nazywa osie, wykres i dodaje legende
def plot_deterministic_loses(episodes, q_det_loses, s_det_loses, dir_name):
    plt.plot(episodes, q_det_loses)
    plt.plot(episodes, s_det_loses)
    plt.xlabel('episode')
    plt.ylabel('deterministic loses')
    plt.title('Loses playing deterministic by Q-learning and SARSA')
    plt.legend(['Q-learning', 'SARSA'])
    plt.savefig(dir_name + '/determ.png')
    plt.close()


# rysuje i zapisuje wykres dla Q-learningu i SARSA ilosci przegranych przy grze strategia
# zachlanna z luka w czasie, nazywa osie, wykres i dodaje legende
def plot_nondeterministic_loses(episodes, q_ndet_loses, s_ndet_loses, dir_name):
    plt.plot(episodes, q_ndet_loses)
    plt.plot(episodes, s_ndet_loses)
    plt.xlabel('episode')
    plt.ylabel('nondeterministic loses')
    plt.title('Loses playing nondeterministic by Q-learning and SARSA')
    plt.legend(['Q-learning', 'SARSA'])
    plt.savefig(dir_name + '/ndeterm.png')
    plt.close()


# rysuje i zapisuje wykres dla Q-learningu i SARSA wielkosci luki
#  w czasie, nazywa osie, wykres i dodaje legende
def plot_gaps(episodes, q_alg_gap, s_alg_gap, dir_name):
    plt.plot(episodes, q_alg_gap)
    plt.plot(episodes, s_alg_gap)
    plt.xlabel('episode')
    plt.ylabel('algorythm gap')
    plt.title('Gap in time by Q-learning and SARSA')
    plt.legend(['Q-learning', 'SARSA'])
    plt.savefig(dir_name + '/gap.png')
    plt.close()


# rysuje i zapisuje wykres dla Q-learningu odsetku przegranych przy grze strategia
# zachlanna z luka oraz rozmiar luki w czasie, nazywa osie, wykres i dodaje legende
def plot_gaps_qlearning(episodes, q_alg_gap, q_ndet_loses, dir_name):
    plt.plot(episodes, q_alg_gap)
    plt.plot(episodes, q_ndet_loses / 200)
    plt.xlabel('episode')
    plt.ylabel('gap/loses part on 200 tries')
    plt.title('Gap and loses in time by Q-learning')
    plt.legend(['gap', 'loses'])
    plt.savefig(dir_name + '/q_gap.png')
    plt.close()


# rysuje i zapisuje wykres dla SARSA odsetku przegranych przy grze strategia
# zachlanna z luka oraz rozmiar luki w czasie, nazywa osie, wykres i dodaje legende
def plot_gaps_sarsa(episodes, s_alg_gap, s_ndet_loses, dir_name):
    plt.plot(episodes, s_alg_gap)
    plt.plot(episodes, s_ndet_loses / 200)
    plt.xlabel('episode')
    plt.ylabel('gap/loses part on 200 tries')
    plt.title('Gap and loses in time by SARSA')
    plt.legend(['gap', 'loses'])
    plt.savefig(dir_name + '/s_gap.png')
    plt.close()


# rysuje wszystkie wykresy
def make_plots(episodes, q_det_loses, s_det_loses, q_ndet_loses, s_ndet_loses, q_alg_gap, s_alg_gap, dir_name):
    os.makedirs(dir_name)
    plot_deterministic_loses(episodes, q_det_loses, s_det_loses, dir_name)
    plot_nondeterministic_loses(episodes, q_ndet_loses, s_ndet_loses, dir_name)
    plot_gaps(episodes, q_alg_gap, s_alg_gap, dir_name)
    plot_gaps_qlearning(episodes, q_alg_gap, q_ndet_loses, dir_name)
    plot_gaps_sarsa(episodes, s_alg_gap, s_ndet_loses, dir_name)


with open('results/results2_1.txt') as input_file:
    for line in input_file:  # petla po wszystkich wierszach w pliku
        nline = line.split()  # podzial wiersza na slowa
        # ponizej nastepuje rozpoznanie momentu opisu an podstawie slow kluczy
        # i zmiennych kontrolujacych
        if nline[0] == "EPSILON":  # parametry dla danego procesu uczenia
            epsilon = float(nline[1])
        elif nline[0] == "ALPHA":
            alpha = float(nline[1])
        elif nline[0] == "GAMMA":
            gamma = float(nline[1])
        elif nline[0] == "Episode:" and i < POINT_NUMBER - 1:  # kolejna walidacja
            i += 1
            episodes[i] = int(nline[1])
        elif nline == ['Deterministic', 'for', 'Q-learning:']:
            # od teraz dane uzyskane z pliku dotyczna deterministyczych gier Q-learningu
            qswitch_det = True
        elif nline[0] == "Loses:" and qswitch_det is True:
            q_det_loses[i] = int(nline[1])
        elif nline[0] == "Algorythm" and qswitch_det is True:
            q_alg_gap[i] = float(nline[2])
        elif nline == ['Nondeterministic', 'for', 'Q-learning:']:
            # od teraz dane uzyskane z pliku dotyczna niedeterministyczych gier Q-learningu
            qswitch_det = False
            qswitch_ndet = True
        elif nline[0] == "Loses:" and qswitch_ndet is True:
            q_ndet_loses[i] = int(nline[1])
        elif nline == ['Deterministic', 'for', 'SARSA:']:
            # od teraz dane uzyskane z pliku dotyczna deterministyczych gier SARSA
            sswitch_det = True
            qswitch_ndet = False
        elif nline[0] == "Loses:" and sswitch_det is True:
            s_det_loses[i] = int(nline[1])
        elif nline[0] == "Algorythm" and sswitch_det is True:
            s_alg_gap[i] = float(nline[2])
        elif nline == ['Nondeterministic', 'for', 'SARSA:']:
            # od teraz dane uzyskane z pliku dotyczna niedeterministyczych gier SARSA
            sswitch_det = False
            sswitch_ndet = True
        elif nline[0] == "Loses:" and sswitch_ndet is True:
            s_ndet_loses[i] = int(nline[1])
        elif nline == ['Deterministic', 'versus', 'games:']:
            # od teraz dane uzyskane z pliku dotyczna deterministyczych gier miedzy uczniami
            sswitch_ndet = False
            det_versus_switch = True
        elif nline[0] == "Ties:" and det_versus_switch is True:
            q_det_loses[i] += int(nline[7])
            s_det_loses[i] += int(nline[4])
        elif nline == ['Nondeterministic', 'versus', 'games']:
            # od teraz dane uzyskane z pliku dotyczna niedeterministyczych gier miedzy uczniami
            det_versus_switch = False
            ndet_versus_switch = True
        elif nline[0] == "Q-learning" and nline[1] == "loses:" and ndet_versus_switch is True:
            q_ndet_loses[i] += int(nline[2])
        elif nline[0] == "Q-learning" and nline[1] == "wins:" and ndet_versus_switch is True:
            s_ndet_loses[i] += int(nline[2])
        elif nline[0] == "========================":
            eq_count += 1
        if eq_count == 3:  # sygnal konca danych dla danego zestawu parametrow - przechodzimy do rysowania wykresow
            dir_name = "plots/EPS:" + "{0:.2f}".format(epsilon) + " ALP:" + "{0:.2f}".format(alpha) + " GAM:" + \
                       "{0:.2f}".format(gamma)  # nazwa folderu na ten zestaw wykresow
            make_plots(episodes, q_det_loses, s_det_loses, q_ndet_loses, s_ndet_loses, q_alg_gap, s_alg_gap, dir_name)
            # zerowanie wszystkich zmiennych przed kolejnym zestawem parametrow
            eq_count = 0
            i = -1
            ndet_versus_switch = False
            episodes = np.zeros(shape=POINT_NUMBER)
            q_det_loses = np.zeros(shape=POINT_NUMBER)
            s_det_loses = np.zeros(shape=POINT_NUMBER)
            q_ndet_loses = np.zeros(shape=POINT_NUMBER)
            s_ndet_loses = np.zeros(shape=POINT_NUMBER)
            q_alg_gap = np.zeros(shape=POINT_NUMBER)
            s_alg_gap = np.zeros(shape=POINT_NUMBER)
