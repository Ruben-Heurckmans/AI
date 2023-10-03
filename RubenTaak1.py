import streamlit as st
from simpleai.search import CspProblem, backtrack

import pandas as pd

#basic var
variables = []
domains = {}

#add_word functie voegt letters toe aan de var variables als deze er nog niet instaan, anders gaat die na of het de eerste letter is van een woord en update het domein
def add_word(word):
    for letter in word:
        if letter not in variables:
            if letter == word[0]:
                domains.update({letter : list(range(1, 10))})
                variables.append(letter)
            else:
                domains.update({letter : list(range(0, 10))})
                variables.append(letter)
        else:
            if letter == word[0]:
                domains.update({letter : list(range(1, 10))})

def constraint_unique(variables, values):
    return len(values) == len(set(values))  # remove repeated values and count

#constraint_add gaat alle waardes na dat de letter kunnen hebben, als de som van woord1 en woord2 gelijk is aan dat van woord3 dan worden die waardes doorgegeven
def constraint_add(variables, values):
    woord1 = word1
    woord2 = word2
    woord3 = word3
    factor1 = ''
    factor2 = ''
    result = ''
    
    #kijkt hoe lang een woord is om dan de waardes in de factor te steken
    for var in range(0,len(woord2)):
        factor1 += str(values[variables.index(woord1[var])])
    factor1 =  int(factor1)

    for var in range(0,len(woord2)):
        factor2 += str(values[variables.index(woord2[var])])
    factor2 =  int(factor2)

    for var in range(0,len(woord3)):
        result += str(values[variables.index(woord3[var])])
    result =  int(result)

    #als de som van factor1 en 2 gelijk zijn aan result wordt de waarde pas doorgegeven en stopt de functie
    return (factor1 + factor2) == result

#main functie
def bereken(): 
    solution = {}
    add_word(word1)
    add_word(word2)
    add_word(word3)

    constraints = [
        (variables, constraint_unique),
        (variables, constraint_add),
    ]

    problem = CspProblem(variables, domains, constraints)
    output = backtrack(problem)

    #overloopt de letters van output en plaatst die in de var antwoord met hun waarde
    antwoord = ''
    for var in output: 
        antwoord += str(var) + " = " + str(output[var]) + ", "
    st.text(antwoord)
    
    #overloopt de letter van een woord om daar de waardes van weer te geven
    w1ant = ''
    for letter in word1:
        w1ant += str(output[letter])
    solution[word1] = w1ant

    w2ant = ''
    for letter in word2:
        w2ant += str(output[letter])
    solution[word2] = w2ant

    w3ant = ''
    for letter in word3:
        w3ant += str(output[letter])
    solution[word3] = w3ant

    #Maak tabel om het mooi weer te geven.
    st.data_editor(
    solution,
    column_config={
        "column 1": "Word",  # change the title
        "column 2": "Value",
    },
    )

st.title("Invoer van 3 waarden")

# Voer de eerste waarde in
word1 = st.text_input("Voer de eerste waarde in:")
# Voer de tweede waarde in
word2 = st.text_input("Voer de tweede waarde in:")
# Voer de derde waarde in
word3 = st.text_input("Voer de derde waarde in:")

# Toon de ingevoerde waarden
st.write(f"De ingevoerde waarden zijn: {word1}, {word2}, {word3}")

#app begint berekening pas als er op de knop gedrukt is
calculate = st.button("Bereken")
if calculate:
    bereken()



