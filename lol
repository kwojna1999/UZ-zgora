#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
from Tkinter import *
from ttk import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import Tkinter as tk
import networkx as nx
import numpy as np
import ScrolledText
from Tkinter import INSERT
from Tkinter import END
from string import maketrans
import unicodedata




win = tk.Tk()
#win.title("COVID - 19 Ulitmate Graph Tool")
#win.iconbitmap('1.ico')
win.geometry('900x400')
win.resizable(0,0)
a=0
b=0
G = nx.Graph()

# Modify adding a Label
aLabel = Label(win, text="Liczba wierzchołków:").grid(column=0, row=0)
aLabel = Label(win, text="Wierzchołek: ").grid(column=0, row=1)
aLabel = Label(win, text="  połącz z wierzchołkiem: ").grid(column=2, row=1)
aLabel = Label(win, text="  Logi: ").grid(column=0, row=2)
aLabel = Label(win, text="  Macierz sąsiedztwa: ").grid(column=1, row=2)
aLabel = Label(win, text="  Macierz incydencji: ").grid(column=2, row=2)
aLabel = Label(win, text="  Lista sąsiedztwa: ").grid(column=3, row=2)
aLabel = Label(win, text="  Liczba kolumn lub wierszy: ").grid(column=1, row=5)

# Button Click Event Function
# Generating combobox data
def clickMe():
    action.configure(text='Dodano ' + lwierzcholkowEntered.get() + ' wierzchołków' )
    Log.insert(INSERT, 'Wierzchołki: ' + lwierzcholkowEntered.get() + '\n')
    lwierzcholkow1 = int(lwierzcholkowEntered.get())
    lwierzcholkow1 = int(lwierzcholkow1)
    getlwierzkolkow = [ i for i in range (0, lwierzcholkow1)]
    #action.configure(text='Dodano ' + lwierzcholkowEntered.get() + ' wierzchołków')
    #number = tk.StringVar()
    #numberChosen = Combobox(win, width=12, textvariable=number)
    #action.configure(text='Dodano ' + getlwierzkolkow.get() + ' wierzchołków')
    numberChosenwierzch1['values'] = getlwierzkolkow
    #numberChosen.grid(column=1, row=1)
    numberChosenwierzch1.current(0)

    numberChosenwierzch2['values'] = getlwierzkolkow
    # numberChosen.grid(column=1, row=1)
    numberChosenwierzch2.current(0)
def reloadbutton():
    braction.configure(text=' Połącz wierzchołki ')

def closeApp():
    win.destroy()

def connectMe():
    braction.configure(text= numberChosenwierzch1.get() + ' połączono z ' + numberChosenwierzch2.get())
    #print numberChosenwierzch1.get()
    #print numberChosenwierzch2.get()
    numberChosenwierzch1_1 = int(numberChosenwierzch1.get())
    numberChosenwierzch1_1 = int(numberChosenwierzch1_1)
    numberChosenwierzch2_2 = int(numberChosenwierzch2.get())
    numberChosenwierzch2_2 = int(numberChosenwierzch2_2 )
    #fegh=[numberChosenwierzch1_1,numberChosenwierzch2_2]
    G.add_edge(numberChosenwierzch1_1,numberChosenwierzch2_2)
    win.after(2000, reloadbutton)
    Log.insert(INSERT, 'Połączenie: ' + numberChosenwierzch1.get() + ' z ' + numberChosenwierzch2.get() + '\n')
    #braction.configure(text='Połącz wierzchołki')
    return G.add_edge(numberChosenwierzch1_1,numberChosenwierzch2_2)


def connectimplement():
    numberChosenwierzch1_1 = int(numberChosenwierzch1)
    numberChosenwierzch2_2 = int(numberChosenwierzch2)

# Generowanie grafu na podstawie połączeń
def openLog():
    lwierzcholkow1 = int(lwierzcholkowEntered.get())
    lwierzcholkow1 = int(lwierzcholkow1)

    getlwierzkolkow = [i for i in range(0, lwierzcholkow1)]
    nodes = getlwierzkolkow
    # global table
    #matrix = table.table.get()
    #matrix = np.array(matrix)
    top = Toplevel()
    top.title("My Second Win")
    #top.iconbitmap('1.ico')
    f = plt.figure(figsize=(5, 4))
    plt.axis('off')

    #edges = ([1, 2], [1, 3], [1, 5], [2, 3], [2, 4], [3, 4], [3, 5], [4, 6], [5, 6])
    #G = nx.DiGraph()

    G.add_nodes_from(nodes)
    #G.add_edges_from(edges)
    #H = nx.Graph(G)
    #G = nx.complete_graph(5)
    print type(G)
    A = nx.to_numpy_matrix(G)
    np.savetxt("am.txt", A, delimiter=' ', fmt='%d')

    C = nx.incidence_matrix(G)
    L = C.toarray()
    Neigh.delete(1.0, END)
    for line in nx.generate_adjlist(G):
        Neigh.insert(INSERT, line + '\n')
    Inci.delete(1.0, END)
    Inci.insert(INSERT, L)
    Adj.delete(1.0, END)
    Adj.insert(INSERT, A)
    print type(Adj)
    print type(A)
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos=pos)
    canvas = FigureCanvasTkAgg(f, master=top)
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.


# Macierz incydencji wprowadzanie zmian
def openAdj():
    #kolumnwierszy = int(KolbWierszEntered.get())
    #kolumnwierszy = int(kolumnwierszy)
    #R=kolumnwierszy
    #C=kolumnwierszy
    #X = np.loadtxt("am.txt")
    #print X
    #print type(X)
    #print len(X[1])
    #Y = np.array(X)#.reshape(R,C)
    #print Y
    #print type(Y)
    #print len(Y[1])

    O = Adj.get(1.0, END)
    print O
    #print type(O)
    #print len(O[1])
    W=np.array(O)
    C=W.tostring()
    print np.fromstring(C, dtype=int)
    print W
    print type(W)
    #print len(W[1])

    #am=nx.to_numpy_matrix(O)
    #AM = np.matrix(O)
   # unicodedata.normalize('NFKD', O).encode('ascii','ignore')
    #print type(O)
    #print (O)
    # O=np.matrix([[0,1],[1,0]])
    #print O
    #d = [[s.encode('ascii') for s in list] for list in O]
    #print type(d)
    #print d

    K = nx.from_numpy_matrix(W)
    top = Toplevel()
    top.title("My Second Win")
    top.iconbitmap('1.ico')
    f = plt.figure(figsize=(5, 4))
    plt.axis('off')

    pos = nx.circular_layout(K)
    nx.draw_networkx(K, pos=pos)
    canvas = FigureCanvasTkAgg(f, master=top)
    canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)  # ERROR Tk.


def openInci():
    b=a

def openNeigh():
    a=b

#Adding a texbox Entry widgety
#lwierzcholkow = tk.StringVar()
lwierzcholkowEntered = Entry(win, width=5)
lwierzcholkowEntered.grid(column=1, row=0)
KolbWierszEntered = Entry(win, width =7)
KolbWierszEntered.grid(column=1, row=6)

# Adding A Buttons
action = Button(win, text="Dodaj", command=clickMe)
action.grid(column=2, row=0)
braction = Button(win, text="Połącz wierzchołki", command=connectMe)
braction.grid(column=4, row=1)
btaction = Button(win, text="Graph Generate from form", command=openLog)
btaction.grid(column=0, row=5)
btaction = Button(win, text="Graph Generate from Adj", command=openAdj)
btaction.grid(column=1, row=7)
btaction = Button(win, text="Graph Generate from Inci", command=openInci)
btaction.grid(column=2, row=5)
btaction = Button(win, text="Graph Generate from List", command=openNeigh)
btaction.grid(column=3, row=5)
actionex = Button(win, text="Egzit", command=closeApp)
actionex.grid(column=2, columnspan=2)

#btaction.pack()
#getlwierzkolkow=0
#Adding A Comboboxes

numberChosenwierzch1 = Combobox(win, width=12, state='readonly')
numberChosenwierzch1.grid(column=1, row=1)
numberChosenwierzch2 = Combobox(win, width=12, state='readonly')
numberChosenwierzch2.grid(column=3, row=1)

# Scrolled Bary
#wrap=tk.WORD, wrap=tk.WORD, state=tk.DISABLED
scrolW = 20
scrolH = 10
Log = ScrolledText.ScrolledText(win, width=scrolW, height=scrolH)
Log.grid(column=0, row=4)
Adj = ScrolledText.ScrolledText(win, width=scrolW, height=scrolH, wrap=tk.WORD)
Adj.grid(column=1, row=4)
Inci = ScrolledText.ScrolledText(win, width=scrolW, height=scrolH)
Inci.grid(column=2, row=4)
Neigh = ScrolledText.ScrolledText(win, width=scrolW, height=scrolH)
Neigh.grid(column=3, row=4)
#b = numberChosenwierzch1.get()
#a = numberChosenwierzch2.get()
#c = a + b
win.mainloop()
