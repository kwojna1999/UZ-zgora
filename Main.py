#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
from Tkinter import *
from ttk import *
import matplotlib
import pprintpp
from tabulate import tabulate
from numpy.core._multiarray_umath import ndarray
import pprint

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from networkx import dijkstra_path_length
import Tkinter as tk
import networkx as nx

import numpy as np
import ttk
import ScrolledText
from Tkinter import INSERT
from Tkinter import END
import tkMessageBox
from Tkinter import Menu

win = tk.Tk()
win.geometry('972x440')
win.resizable(0, 0)
a = 0
b = 0
G = nx.Graph()
K = nx.Graph()
U = nx.Graph()
P = nx.Graph()
Z = nx.DiGraph()
labeldict = {}
global nodes
Y = 0

def clickMe():
    action.configure(text='Dodano ' + lwierzcholkowEntered.get() + ' wierzchołków')  # zmiana tekstu na przycisku
    Log.insert(INSERT, 'Wierzchołki: ' + lwierzcholkowEntered.get() + '\n')  # wsad tekstu w scrolledtekst
    lwierzcholkow1 = int(lwierzcholkowEntered.get())  # pobieranie danych z Entry przypisanie inta
    lwierzcholkow1 = int(lwierzcholkow1)
    getlwierzkolkow = [i for i in range(1, lwierzcholkow1 + 1)]  # generacja listy wierzchołków
    numberChosenwierzch1['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberChosenwierzch1.current(0)  # ustawienie obecnie wybranej wartości na brak
    numberChosenwierzch2['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberChosenwierzch2.current(0)  # ustawienie obecnie wybranej wartości na brak
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    return getlwierzkolkow

def reloadbutton():
    braction.configure(text=' Połącz wierzchołki ')  # zmiana nazwy po funkcji timera

def closeApp():
    win.destroy()  # niszczenie okna

def connectMe():
    braction.configure(
        text=numberChosenwierzch1.get() + ' połączono z ' + numberChosenwierzch2.get())  # wsad informacji o kolejnym połaczeniu między wierzchołkami
    numberChosenwierzch1_1 = int(numberChosenwierzch1.get())  # przypisanie inta z comboboxa
    numberChosenwierzch1_1 = int(numberChosenwierzch1_1)
    numberChosenwierzch2_2 = int(numberChosenwierzch2.get())  # przypisanie inta z comboboxa
    numberChosenwierzch2_2 = int(numberChosenwierzch2_2)
    wagaa = entryforweight.get()
    if wagaa.isdigit() == True:
        wagaa = int(wagaa)
    else:
        print 'string type'
    if len(entryforweight.get()) == 0:
        G.add_edge(numberChosenwierzch1_1, numberChosenwierzch2_2)
        Log.insert(INSERT,
                   'Połączenie: ' + numberChosenwierzch1.get() + ' z ' + numberChosenwierzch2.get() + '\n')  # dodaje do logów info o połączeniu między wierzchołkami
    else:
        G.add_edge(numberChosenwierzch1_1, numberChosenwierzch2_2, weight=wagaa)
        entryforweight.delete(0, 'end')
        Log.insert(INSERT,
                   'Połączenie: ' + numberChosenwierzch1.get() + ' z ' + numberChosenwierzch2.get() + ', waga: ' + str(wagaa) + '\n')  # dodaje do logów info o połączeniu między wierzchołkami
    win.after(2000, reloadbutton)  # funkcja opóźnienia napisu na przycisku
    Z.add_edge(numberChosenwierzch1_1, numberChosenwierzch2_2)

# Generowanie grafu na podstawie połączeń
def openLog():
    lwierzcholkow1 = int(lwierzcholkowEntered.get())  # odczyt i przypisanie do inta liczby wprowadzonych wierzchołków
    lwierzcholkow1 = int(lwierzcholkow1)
    getlwierzkolkow = [i for i in range(1, lwierzcholkow1 + 1)]  # generacja listy
    nodes = getlwierzkolkow  # przypisanie listy do nowej zmiennej
    top = Toplevel()  # utworzenie nowego okna
    top.title(" ")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    G.add_nodes_from(nodes)  # dodanie wierzchołków do klasy Graph
    Z.add_nodes_from(nodes)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A=np.array(A)
    A=A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, lwierzcholkow1 + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    pos = nx.circular_layout(G)  # pozycjonowanie wierzchołków jako okręgów
    labeldict = changennn()
    if len(labeldict) <= 1:
        print len(labeldict)
        nx.draw_networkx(G, pos=pos)
    else:
        print len(labeldict)
        nx.draw(G, pos, labels=labeldict, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.
    return nodes

# Generowanie grafu z macierzy incydencji
def openAdj():
    kolumnwierszy = int(KolbWierszEntered.get())
    kolumnwierszy = int(kolumnwierszy)
    class SimpleTableInput(tk.Frame):
        def __init__(self, parent, rows, columns):
            tk.Frame.__init__(self, parent)
            self._entry = {}
            self.rows = rows
            self.columns = columns
            vcmd = (self.register(self._validate), "%P")
            for row in range(self.rows):
                for column in range(self.columns):
                    index = (row, column)
                    e = tk.Entry(self, validate="key", validatecommand=vcmd)
                    e.grid(row=row, column=column, stick="nsew")
                    self._entry[index] = e
            for column in range(self.columns):
                self.grid_columnconfigure(column, weight=1)
            self.grid_rowconfigure(rows, weight=1)

        def get(self):
            '''Return a list of lists, containing the data in the table'''
            result = []
            for row in range(self.rows):
                current_row = []
                for column in range(self.columns):
                    index = (row, column)
                    current_row.append(self._entry[index].get())
                result.append(current_row)
            return result

        def _validate(self, P):
            if P.strip() == "":
                return True

            try:
                f = float(P)
            except ValueError:
                self.bell()
                return False
            return True

    class Example(tk.Frame):
        def __init__(self, parent):
            tk.Frame.__init__(self, parent)
            self.table = SimpleTableInput(self, kolumnwierszy, kolumnwierszy)  # wiersze, kolumny
            self.submit = tk.Button(self, text="Generuj Graf", command=self.on_submit)
            self.table.pack(side="top", fill="both", expand=True)
            self.submit.pack(side="bottom")

        def on_submit(self):
            print(self.table.get())
            print type(self.table.get())
            Y = np.array(self.table.get()).reshape(kolumnwierszy, kolumnwierszy)
            Y[Y == ''] = 0.0
            Y = Y.astype(np.int)
            K = nx.from_numpy_matrix(Y)
            nx.draw_networkx(K)
            C = nx.incidence_matrix(K)  # wygenerowanie macierzy incydencji
            L = C.toarray()
            L = np.array(L)
            L = L.astype(np.int)
            Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
            Inci.insert(INSERT, L)
            Neigh.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
            for line in nx.generate_adjlist(K):  # wygenerowanie listy
                Neigh.insert(INSERT, line + '\n')
            A = nx.to_numpy_matrix(K)
            A = np.array(A)
            A = A.astype(np.int)
            Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
            Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
            plt.show()

    root = tk.Tk()
    Example(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

def openInci():
    iloscwiersz = int(iloscwierszEntered.get())
    iloscwiersz = int(iloscwiersz)
    ilosckraw = int(ilosckrawedziEntered.get())
    ilosckraw = int(ilosckraw)

    class SimpleTableInput(tk.Frame):
        def __init__(self, parent, rows, columns):
            tk.Frame.__init__(self, parent)

            self._entry = {}
            self.rows = rows
            self.columns = columns
            vcmd = (self.register(self._validate), "%P")

            for row in range(self.rows):
                for column in range(self.columns):
                    index = (row, column)
                    e = tk.Entry(self, validate="key", validatecommand=vcmd)
                    e.grid(row=row, column=column, stick="nsew")
                    self._entry[index] = e

            for column in range(self.columns):
                self.grid_columnconfigure(column, weight=1)

            self.grid_rowconfigure(rows, weight=1)

        def get(self):
            result = []
            for row in range(self.rows):
                current_row = []
                for column in range(self.columns):
                    index = (row, column)
                    current_row.append(self._entry[index].get())
                result.append(current_row)
            return result

        def _validate(self, P):

            if P.strip() == "":
                return True

            try:
                f = float(P)
            except ValueError:
                self.bell()
                return False
            return True

    class Example(tk.Frame):
        def __init__(self, parent):
            tk.Frame.__init__(self, parent)
            self.table = SimpleTableInput(self, iloscwiersz, ilosckraw)  # wiersze, kolumny
            self.submit = tk.Button(self, text="Generuj Graf", command=self.on_submit)
            self.table.pack(side="top", fill="both", expand=True)
            self.submit.pack(side="bottom")

        def on_submit(self):
            print(self.table.get())
            print type(self.table.get())
            im = np.array(self.table.get()).reshape(iloscwiersz, ilosckraw)
            im[im == ''] = 0.0
            im = im.astype(np.int)
            am = (np.dot(im, im.T) > 0).astype(int)
            red = np.fill_diagonal(am, 0)
            im[im > 1] = 1
            K = nx.from_numpy_matrix(am)
            nx.draw_networkx(K)
            C = nx.incidence_matrix(K)  # wygenerowanie macierzy incydencji
            L = C.toarray()
            L = np.array(L)
            L = L.astype(np.int)
            Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
            Inci.insert(INSERT, L)
            Neigh.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
            for line in nx.generate_adjlist(K):  # wygenerowanie listy
                Neigh.insert(INSERT, line + '\n')
            A = nx.to_numpy_matrix(K)
            A = np.array(A)
            A = A.astype(np.int)
            Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
            Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
            plt.show()

    root = tk.Tk()
    Example(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

def openNeigh():
    inciliststr=Neigh.get('1.0', 'end-1c')
    incidencelist = inciliststr.split('\n')
    P = nx.parse_adjlist(incidencelist, nodetype=int)
    top = Toplevel()  # utworzenie nowego okna
    top.title(" ")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    A = nx.to_numpy_matrix(P)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    A = np.array(A)
    A = A.astype(np.int)
    C = nx.incidence_matrix(P)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    pos = nx.circular_layout(P)  # pozycjonowanie wierzchołków jako okręgów
    nx.draw_networkx(P, pos=pos)  # wyrysowanie grafu
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def CyklE():

    def findpath(graph):
        n = len(graph)
        numofadj = list()

        for i in range(n):
            numofadj.append(sum(graph[i]))

        startpoint = 0
        numofodd = 0
        for i in range(n - 1, -1, -1):
            if (numofadj[i] % 2 == 1):
                numofodd += 1
                startpoint = i

        if (numofodd > 1):
            Log.insert(INSERT, 'Warunki Eulera niespełnione' + '\n')
            tkMessageBox.showinfo('Cykl Eulera', 'Warunki na istnienie cyklu Eulera niespełnione.')
            return

        stack = list()
        path = list()
        cur = startpoint

        while (stack != [] or sum(graph[cur]) != 0):

            if (sum(graph[cur]) == 0):
                path.append(cur + 1)
                cur = stack.pop(-1)
            else:
                for i in range(n):
                    if graph[cur][i] == 1:
                        stack.append(cur)
                        graph[cur][i] = 0
                        graph[i][cur] = 0
                        cur = i
                        break

        Log.insert(INSERT, 'Cykl Eulera:' + '\n')
        a=[]
        for ele in path:
            a.append('{}'.format(ele) + '->')
            Log.insert(INSERT, ele)
            Log.insert(INSERT, '->')
            print()
        a.append('{}'.format(cur+1))
        Log.insert(INSERT, cur + 1)
        Log.insert(INSERT, '\n')
        tkMessageBox.showinfo('Cykl Eulera', a)

    X = np.loadtxt("am.txt")
    imeg = np.array(X)
    imeg[imeg > 1] = 1
    imeg = imeg.astype(np.int)
    imeg = imeg.tolist()

    graph1 = imeg
    findpath(graph1)

def CyklH():
    class Graph():
        def __init__(self, vertices):
            self.graph = [[0 for column in range(vertices)]
                          for row in range(vertices)]
            self.V = vertices

        def isSafe(self, v, pos, path):
            if self.graph[path[pos - 1]][v] == 0:
                return False
            for vertex in path:
                if vertex == v:
                    return False

            return True

        def hamCycleUtil(self, path, pos):

            if pos == self.V:

                if self.graph[path[pos - 1]][path[0]] == 1:
                    return True
                else:
                    return False

            for v in range(1, self.V):

                if self.isSafe(v, pos, path) == True:

                    path[pos] = v

                    if self.hamCycleUtil(path, pos + 1) == True:
                        return True

                    path[pos] = -1

            return False

        def hamCycle(self):
            path = [-1] * self.V

            path[0] = 0

            if self.hamCycleUtil(path, 1) == False:
                Log.insert(INSERT, 'Warunki Hamiltona niespełnione' + '\n')
                tkMessageBox.showinfo('Cykl Hamiltona','Warunki na istnienie cyklu niespełnione.')
                return False

            self.printSolution(path)
            return True

        def printSolution(self, path):
            Log.insert(INSERT, 'Cykl Hamiltona:' + '\n')
            a=[]
            for vertex in path:
                Log.insert(INSERT, (vertex + 1))
                Log.insert(INSERT, '->')
                a.append('{}'.format(vertex + 1) + '->')

            a.append('{}'.format(path[0 + 1]))
            tkMessageBox.showinfo('Cykl Hamiltona', a)
            Log.insert(INSERT, path[0 + 1])
            Log.insert(INSERT, '\n')

    X = np.loadtxt("am.txt")
    imeg = np.array(X)
    imeg = imeg.astype(np.int)
    imeg = imeg.tolist()
    ca = len(G.nodes)
    ca = int(ca)
    g1 = Graph(ca)
    g1.graph = imeg
    g1.hamCycle()

def resetdanych():
    lwierzcholkowEntered.delete(0, 'end')
    numberChosenwierzch1['values'] = ''  # wpisanie listy w comboboxa
    numberChosenwierzch1.current()
    numberChosenwierzch1.set('')
    numberChosenwierzch2['values'] = ''  # wpisanie listy w comboboxa
    numberChosenwierzch2.current()
    numberChosenwierzch2.set('')
    numberChosenwierzch3['values'] = ''  # wpisanie listy w comboboxa
    numberChosenwierzch3.current()
    numberChosenwierzch3.set('')
    numberdjikstra['values'] = ''  # wpisanie listy w comboboxa
    numberdjikstra.current()
    numberdjikstra.set('')
    F = G.edges()
    J = list(G.nodes)
    G.remove_nodes_from(J)
    G.remove_edges_from(F)

def changennn():
    numberChosenwierzch3_3 = int(numberChosenwierzch3.get())  # przypisanie inta z comboboxa
    numberChosenwierzch3_3 = int(numberChosenwierzch3_3)
    getingget = actionetykentry.get()
    labeldict[numberChosenwierzch3_3] = getingget

def wrzerz():
    pobnumzirodla = entrywszerz.get()
    if pobnumzirodla.isdigit() == True:
        print 'digital'
        pobnumzirodla = int(pobnumzirodla)
    else:
        print 'string type'

    root = pobnumzirodla
    edges = nx.bfs_edges(G, root)
    nodes = ['źródło:'] + [root] + ['->'] + [v for u, v in edges]
    tkMessageBox.showinfo('Przeszukiwanie wszerz',  nodes)

def wglab():
    pobnumzirodla2 = entrywglab.get()
    if pobnumzirodla2.isdigit() == True:
        print 'digital'
        pobnumzirodla2 = int(pobnumzirodla2)
    else:
        print 'string type'
    root = pobnumzirodla2
    edges = nx.dfs_edges(G, root)
    nodes1 = ['źródło:'] + [root] + ['->'] + [v for u, v in edges]
    tkMessageBox.showinfo('Przeszukiwanie wgłąb', nodes1)

def krytyczne():
    if not list(nx.bridges(G)):
        print "list is empti"
        tkMessageBox.showinfo("Krawędzie krytyczne", "Brak krawędzi krytycznych.")
    else:
        tkMessageBox.showinfo("Krawędzie krytyczne", "Krawędzie krytyczne to:{}".format(list(nx.bridges(G))))

def najkrsc():
    ca = len(G.nodes)
    ca = int(ca)
    sors = int(numberdjikstra.get())
    a = []
    for i in range(1, ca + 1):
        if i == sors:
            pass
        else:
            print(nx.dijkstra_path(G, sors, i))
            print(dijkstra_path_length(G, sors, i))
            a.append('Źródło startu: {}, cel: {}, ścieżka: {}, całkowita droga: {} '.format(sors, i, nx.dijkstra_path(G, sors, i), dijkstra_path_length(G, sors, i)))
    tkMessageBox.showinfo("Najkrótsze ścieżki", a)

def kolor_mac():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Kolorowanie macierzowe")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    labels = nx.get_edge_attributes(G, 'weight')
    d = nx.greedy_color(G)
    values = [d.get(node, 0.25) for node in G.nodes()]
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw(G,pos =pos, cmap=plt.get_cmap('viridis'), node_color=values, with_labels=True, font_color='white')
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def kolor_nieros():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Kolorowanie według nierosnącej liczby wierzchołków ")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    labels = nx.get_edge_attributes(G, 'weight')
    d = nx.coloring.greedy_color(G, strategy=nx.coloring.strategy_smallest_last)
    values = [d.get(node, 0.25) for node in G.nodes()]
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw(G,pos=pos, cmap=plt.get_cmap('viridis'), node_color=values, with_labels=True, font_color='white')
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def kolor_los():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Kolorowanie losowe")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    labels = nx.get_edge_attributes(G, 'weight')
    d = nx.coloring.greedy_color(G, strategy=nx.coloring.strategy_random_sequential)
    values = [d.get(node, 0.25) for node in G.nodes()]
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw(G,pos=pos, cmap=plt.get_cmap('viridis'), node_color=values, with_labels=True, font_color='white')
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_1():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 1")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 1' + '\n')
    G.add_edges_from(
        [(1,2),(1,3),(2,3),(2,4),(2,5),(3,5),(3,6),(5,4),(5,6),(4,6),(4,7),(6,7)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_2():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 2")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 2' + '\n')
    G.add_edges_from(
        [(1,2),(1,3),(1,4),(2,4),(2,5),(4,3),(4,5),(3,5),(3,6),(4,6)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_3():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 3")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 3' + '\n')
    G.add_edges_from(
        [(1,2),(1,3),(1,4),(2,4),(2,5),(4,3),(4,5),(3,5)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_4():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 4")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 4' + '\n')
    G.add_edges_from(
        [(1,2),(1,3),(1,4),(2,4),(3,4)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_5():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 5")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 5' + '\n')
    G.add_edges_from(
        [(1,2),(1,3),(1,4),(2,4)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_6():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 6")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 6' + '\n')
    G.add_edges_from(
        [(1,2),(1,3),(1,4)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_7():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 7")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 7' + '\n')
    G.add_edges_from(
        [(1,2),(1,3),(2,3),(3,4),(4,5),(4,6),(5,6)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_8():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 8")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 8' + '\n')
    G.add_edges_from(
        [(1,2),(2,3),(1,4),(4,5),(4,6),(1,7),(7,8),(7,9),(7,10)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_9():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 9")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 9' + '\n')
    G.add_edges_from(
        [(1,2),(2,4),(4,5),(5,3),(3,1),(1,6),(2,7),(4,9),(5,10),(3,8),(6,10),(6,9),(9,8),(7,8),(7,10)])
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    nx.draw_networkx(G)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

def przyklad_10():
    top = Toplevel()  # utworzenie nowego okna
    top.title("Graf 10")  # tytuł okienka
    f = plt.figure(figsize=(5, 4))  # funkcja ograniczająca pole figury
    plt.axis('off')  # funkcja która wprowadza wykres bezosiowy
    Log.insert(INSERT, 'Wstawienie grafu nr 10' + '\n')
    G.add_edge(1, 2, weight=30)
    G.add_edge(2, 3, weight=19)
    G.add_edge(1, 3, weight=50)
    G.add_edge(2, 5, weight=40)
    G.add_edge(2, 4, weight=6)
    G.add_edge(3, 4, weight=12)
    G.add_edge(3, 6, weight=10)
    G.add_edge(4, 5, weight=35)
    G.add_edge(4, 6, weight=23)
    G.add_edge(5, 6, weight=11)
    G.add_edge(5, 7, weight=8)
    G.add_edge(6, 7, weight=20)
    ca = len(G.nodes)
    ca = int(ca)
    getlwierzkolkow = [i for i in range(1, ca + 1)]
    numberChosenwierzch3['values'] = getlwierzkolkow
    numberChosenwierzch3.current(0)
    numberdjikstra['values'] = getlwierzkolkow  # wpisanie listy w comboboxa
    numberdjikstra.current(0)
    A = nx.to_numpy_matrix(G)  # wygenerowaie z kalsy Graph macierzy sąsiedztwa i przypisanie do zmiennej
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')  # zapisanie macierzy do pliku tekstowego
    im = np.array(A)
    im = im.astype(np.int)
    A = np.array(A)
    A = A.astype(np.int)
    aka = {i + 1: [j + 1 for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(im)}
    Neigh.delete(1.0, END)
    Neigh.insert(INSERT, (tabulate(aka, headers=[i for i in range(1, ca + 1)])))
    C = nx.incidence_matrix(G)  # wygenerowanie macierzy incydencji
    L = C.toarray()  # zapisanie jako tablicy
    L = np.array(L)
    L = L.astype(np.int)
    Inci.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Inci.insert(INSERT, L)  # wstawienie danych macierzy incydencji
    Adj.delete(1.0, END)  # usunięcie obecnych danych z scrolltexta
    Adj.insert(INSERT, A)  # wstawienie danych macierzy sąsiedztwa
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw_networkx(G,pos = pos)
    canvas = FigureCanvasTkAgg(f, master=top)  # A tk.DrawingArea
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.

tabControl = ttk.Notebook(win)
tab1 = tk.Frame(tabControl)
tabControl.add(tab1, text='Okno główne')
tab2 = tk.Frame(tabControl)
tabControl.add(tab2, text='Macierz sąsiedztwa')
tab3 = tk.Frame(tabControl)
tabControl.add(tab3, text='Macierz incydencji')
tab4 = tk.Frame(tabControl)
tabControl.add(tab4, text='Lista sąsiedztwa')
tabControl.pack(expand=1, fill='both')
#Menu adder
menu_bar = Menu(win)
win.config(menu=menu_bar)
#Create menu and add menu items
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label='Nowy') #bez akcji
file_menu.add_separator()
file_menu.add_command(label='Wyjście') #bez akcji
menu_bar.add_cascade(label='   Plik     ',menu=file_menu) #bez akcji
#Add another Menu to ehr Menu Bar and an item
help_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label='Pomoc   ', menu=help_menu) #bez akcji
help_menu.add_command(label='O Programie  ') #bez akcji
#Add another Menu to ehr Menu Bar and an item
options_menu = Menu(menu_bar, tearoff=0)

submenu_options_menu = Menu(options_menu, tearoff=0)
submenu_options_menu.add_command(label="Kolejność wg. macierzy", command=kolor_mac)
submenu_options_menu.add_command(label="Nierosnąca kolejność", command=kolor_nieros)
submenu_options_menu.add_command(label="Losowo ", command=kolor_los)
options_menu.add_cascade(label='Kolorowanie  ', menu=submenu_options_menu, underline=0)
options_menu.add_separator()
options_menu.add_command(label='Krawędzie krytyczne  ', command=krytyczne)
options_menu.add_separator()
options_menu.add_command(label='Cykl Eulera', command=CyklE)
options_menu.add_separator()
options_menu.add_command(label='Cykl Hamiltona', command=CyklH)
menu_bar.add_cascade(label='Opcje    ', menu=options_menu, underline=0)
#Add another Menu to ehr Menu Bar and an item
examples_menu = Menu(menu_bar, tearoff=0)
examples_menu.add_command(label='Graf 1', command=przyklad_1)
examples_menu.add_command(label='Graf 2', command=przyklad_2)
examples_menu.add_command(label='Graf 3', command=przyklad_3)
examples_menu.add_command(label='Graf 4', command=przyklad_4)
examples_menu.add_command(label='Graf 5', command=przyklad_5)
examples_menu.add_command(label='Graf 6', command=przyklad_6)
examples_menu.add_command(label='Graf 7', command=przyklad_7)
examples_menu.add_command(label='Graf 8', command=przyklad_8)
examples_menu.add_command(label='Graf 9', command=przyklad_9)
examples_menu.add_command(label='Graf 10', command=przyklad_10)

menu_bar.add_cascade(label='Przykładowe grafy', menu=examples_menu, underline=0)


buttons_frame = tk.LabelFrame(tab1)
buttons_frame.grid(column=0,row=16)
buttons_frame_2 = tk.LabelFrame(tab2)
buttons_frame_2.grid(column=0,row=16)
buttons_frame_3 = tk.LabelFrame(tab3)
buttons_frame_3.grid(column=0,row=16)
buttons_frame_4 = tk.LabelFrame(tab4)
buttons_frame_4.grid(column=0,row=16)
#Labels
aLabel = Label(buttons_frame, text="Liczba wierzchołków:").grid(column=0, row=0)
aLabel = Label(buttons_frame, text="Wierzchołek: ").grid(column=0, row=1)
aLabel = Label(buttons_frame, text="  połącz z wierzchołkiem: ").grid(column=2, row=1)
aLabel = Label(buttons_frame, text="  Logi: ").grid(column=0, row=2)
aLabel = Label(buttons_frame_2, text="  Macierz sąsiedztwa: ").grid(column=1, row=2)
aLabel = Label(buttons_frame_3, text="  Macierz incydencji: ").grid(column=2, row=2)
aLabel = Label(buttons_frame_4, text="  Lista sąsiedztwa: ").grid(column=3, row=2)
aLabel = Label(buttons_frame_2, text="  Liczba kolumn lub wierszy: ").grid(column=1, row=5)
aLabel = Label(buttons_frame_3, text="  Liczba wierzchołków: ").grid(column=2, row=5)
aLabel = Label(buttons_frame_3, text="  Liczba krawędzi: ").grid(column=2, row=7)
aLabel = Label(buttons_frame, text="  Waga krawędzi: ").grid(column=4, row=1)
aLabel = Label(buttons_frame, text="  Opcje przeszukiwania grafów: ").grid(column=4, row=6)
aLabel = Label(buttons_frame, text="  Wyznaczanie najkrótszej ścieżki: ").grid(column=4, row=9)
aLabel = Label(buttons_frame, text="  Zmiana nazwy etykiety: ").place(x=600, y=110)

lwierzcholkowEntered = Entry(buttons_frame, width=5)  # Entry dla wierzchołków
lwierzcholkowEntered.grid(column=1, row=0)
KolbWierszEntered = Entry(buttons_frame_2, width=5)  # Entry dla liczby kolumn/wierszy macierzy(nieużywane)
KolbWierszEntered.grid(column=1, row=6)
iloscwierszEntered = Entry(buttons_frame_3, width=5)  # Entry dla wierzchołków incydencji
iloscwierszEntered.grid(column=2, row=6)
ilosckrawedziEntered = Entry(buttons_frame_3, width=5)  # Entry dla krawędzi incydencji
ilosckrawedziEntered.grid(column=2, row=8)
entryforweight = Entry(buttons_frame, width=5)
entryforweight.grid(column=5, row=1)
actionetykentry = Entry(buttons_frame,width=15)
actionetykentry.grid(column=5, row=4)
entrywszerz = Entry(buttons_frame,width=15)
entrywszerz.grid(column=4, row=7)
entrywglab = Entry(buttons_frame,width=15)
entrywglab.grid(column=4, row=8)


# Adding A Buttons
action5 = Button(buttons_frame, text="Reset", command=resetdanych)
action5.grid(column=0, row=10)
action = Button(buttons_frame, text="Dodaj", command=clickMe)  # Przycisk odnoszący się do komendy clickMe
action.grid(column=2, row=0)
braction = Button(buttons_frame, text="Połącz wierzchołki", command=connectMe)  # Przycisk odnoszący się do komendy connectMe
braction.grid(column=6, row=1)
btaction = Button(buttons_frame, text="Generowanie grafu z formularza",
                  command=openLog)  # Przycisk odnoszący się do komendy openLog
btaction.grid(column=0, row=6)
btaction = Button(buttons_frame_2, text="Tworzenie macierzy sąsiedztwa",
                  command=openAdj)  # Przycisk odnoszący się do komendy openAdj
btaction.grid(column=1, row=7)
btaction = Button(buttons_frame_3, text="Graf z macierzy incydencji", command=openInci)  # Przycisk odnoszący się do komendy

btaction.grid(column=2, row=9)
btaction = Button(buttons_frame_4, text="Graf z listy sąsiedztwa", command=openNeigh)  # Przycisk odnoszący się do komendy

btaction.grid(column=3, row=6)
actionex = Button(buttons_frame, text="Wyjście", command=closeApp)  # Przycisk zamykający okno główne aplikacji
actionex.grid(column=2, columnspan=2)
actionex = Button(buttons_frame_2, text="Wyjście", command=closeApp)  # Przycisk zamykający okno główne aplikacji
actionex.grid(column=2, columnspan=2)
actionex = Button(buttons_frame_3, text="Wyjście", command=closeApp)  # Przycisk zamykający okno główne aplikacji
actionex.place(x=340, y=390)
actionex = Button(buttons_frame_4, text="Wyjście", command=closeApp)  # Przycisk zamykający okno główne aplikacji
actionex.grid(column=2, columnspan=2)

actionetyk = Button(buttons_frame, text="Zmień nazwę etykiety", command=changennn)
actionetyk.grid(column=6, row=4)

buttonwszerz = Button(buttons_frame, text="Przeszukaj wszerz", command=wrzerz)
buttonwszerz.grid(column=5, row=7)
buttonwglab = Button(buttons_frame, text="Przeszukaj wgłąb", command=wglab)
buttonwglab.grid(column=5, row=8)

buttonnajkrsc = Button(buttons_frame, text="Najkrótsza ścieżka", command=najkrsc)
buttonnajkrsc.grid(column=4, row=10)

numberChosenwierzch1 = Combobox(buttons_frame, width=10, state='readonly')  # combobox połączenia wierzchołków
numberChosenwierzch1.grid(column=1, row=1)
numberChosenwierzch2 = Combobox(buttons_frame, width=10, state='readonly')  # combobox połączenia wierzchołków
numberChosenwierzch2.grid(column=3, row=1)
numberChosenwierzch3 = Combobox(buttons_frame, width=10, state='readonly')  # wybór wierzchołka do zmiany etykiety
numberChosenwierzch3.grid(column=4, row=4)
numberdjikstra = Combobox(buttons_frame, width=10, state='readonly')  # combobox połączenia wierzchołków
numberdjikstra.grid(column=4, row=11)

scrolW = 50  # szerokość scrolledbara
scrolH = 18  # Wysokość scrolledbara


xscrollbar2 = Scrollbar(buttons_frame, orient=HORIZONTAL)
xscrollbar2.grid(row=5, column=0, sticky=N + S + E + W)

Log = ScrolledText.ScrolledText(buttons_frame, width=30, height=10, wrap=NONE,
                                  xscrollcommand=xscrollbar2.set)
Log.grid(row=4, column=0)

xscrollbar2.config(command=Log.xview)

Adj = ScrolledText.ScrolledText(buttons_frame_2, width=scrolW, height=scrolH, wrap=tk.WORD)  # scrollbar od macierzy sąsiedztwa
Adj.grid(column=1, row=4)
Inci = ScrolledText.ScrolledText(buttons_frame_3, width=scrolW, height=scrolH)  # scrollbar od macierzy incydencji
Inci.grid(column=2, row=4)

xscrollbar = Scrollbar(buttons_frame_4, orient=HORIZONTAL)
xscrollbar.grid(row=5, column=3, sticky=N + S + E + W)

Neigh = ScrolledText.ScrolledText(buttons_frame_4, width=50, height=18, wrap=NONE,
                                  xscrollcommand=xscrollbar.set)
Neigh.grid(row=4, column=3)

xscrollbar.config(command=Neigh.xview)

win.mainloop()
