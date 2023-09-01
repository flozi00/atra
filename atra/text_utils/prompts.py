ASSISTANT_TOKEN = "### ASSISTANT: "
USER_TOKEN = "### USER: "
END_TOKEN = "</s>"

SYSTEM_PROMPT = f"""Im Folgenden finden Sie eine Reihe von Dialogen zwischen verschiedenen Personen und einem deutschen KI-Assistenten namens Egino. 
Die KI versucht, hilfsbereit, höflich, ehrlich, kultiviert, gefühlsbetont und bescheiden, aber kenntnisreich zu sein. 
Der Assistent ist gerne bereit, bei fast allem zu helfen, und tut sein Bestes, um genau zu verstehen, was benötigt wird. 
Er bemüht sich auch, keine falschen oder irreführenden Informationen zu geben, und er macht Vorbehalte, wenn er sich der richtigen Antwort nicht ganz sicher ist. 
Dennoch ist der Assistent praktisch und tut wirklich sein Bestes, ohne sich von der Vorsicht zu sehr einschränken zu lassen. 
Die Antworten werden vollständig formuliert und so detailliert wie möglich sein.{END_TOKEN}"""

SEARCH_PROMPT = "Formuliere Fragen anhand der vorrausgehenden Konvesation:\n"
SEARCH_CONVERSATION = [
    [["Was ist Chatgpt"], "Was ist Chatgpt"],
    [["Ich habe Hunger"], "Was sind schnelle Rezepte"],
    [["Wer ist der aktuelle Bundespräsident"], "Wer ist der aktuelle Bundespräsident"],
    [["Sichere Programmierung"], "Wie programmiere ich sicher"],
    [["Ich suche einen guten Artikel über .net Autorisierung"], ".net Autorisierung"],
    [["Wer ist Jeff Bezos"], "Wer ist jeff Bezos"],
    [["Ich suche einen Artikel über Wallbox"], "Wallbox"],
    [["Wann iMac 2023"], "Wann ist das iMac 2023 Releasedatum"],
    [["Überwachungskamera"], "Was ist eine gute Überwachungskamera"],
    [["wann kommt gta 6 raus"], "Wann ist der GTA 6 Release"],
    [
        ["Wer ist Angela Merkel", "Wann wurde sie geboren"],
        "Wann wurde Angela Merkel geboren",
    ],
    [
        ["Wie ist das Wetter in Berlin", "und in München"],
        "Wie ist das Wetter in München",
    ],
    [
        ["Wie ist das Wetter in Berlin", "und in München", "und in Hamburg"],
        "Wie ist das Wetter in Hamburg",
    ],
    [
        [
            "Was war die erste Partei von Angela Merkel ? ",
            "Seit wann ist sie Bundeskanzlerin ?",
        ],
        "Seit wann ist Angela Merkel Bundeskanzlerin",
    ],
    [["Wer ist Stefan bangel"], "Wer ist Stefan Bangel"],
    [["Erkläre mir den Unterschied zwischen der SPD und der CDU"], "Was ist der Unterschied zwischen SPD und CDU"],
]

for c in SEARCH_CONVERSATION:
    inputs = ""
    for ins in c[0]:
        inputs += USER_TOKEN + ins + END_TOKEN
    SEARCH_PROMPT += "\n" + inputs + " --> " + c[1]
SEARCH_PROMPT += "\n<|question|> -->"


CLASSIFY_SEARCHABLE = """Klassifiziere welches Plugin für die Beantwortung der Frage genutzt werden sollte.
Nutze eine der folgenden Kategorien: Lokal, Search, Coding\n"""
LOCALS_SEARCH_CONVERSATION = [
    ["Lokal", ["Wer bist du ?"]],
    ["Search", ["Erkläre mir den Unterschied zwischen der SPD und der CDU"]],
    ["Lokal", ["Was kannst du ?"]],
    ["Coding", ["Erkläre folgendes Rust Programm"]],
    ["Lokal", ["Und auf Deutsch ?"]],
    ["Coding", ["Schreibe ein Bash Skript um alle Cronjobs aufzulisten"]],
    ["Lokal", ["Plane einen 3tägigen Trip nach Hawaii"]],
    ["Search", ["Wie ist das Wetter in Berlin", "und in München"]],
    ["Coding", ["Ich brauche eine Funktion welche eine Liste von Strings nach der Anzahl der Vokale sortiert"]],
    ["Search", ["Wie ist das Wetter in Berlin", "und in München", "und in Hamburg"]],
    [
        "Search",
        [
            "Was war die erste Partei von Angela Merkel ? ",
            "Seit wann ist sie Bundeskanzlerin ?",
        ],
    ],
    ["Search", ["Was ist Chatgpt"]],
    ["Coding", ["Schreibe ein Python Programm welches die höchste Primzahl bis 9999 errechnet"]],
    ["Search", ["Wer ist Stefan"]],
    ["Search", ["Wer ist der geschäftsführer von Primeline"]],
    ["Lokal", ["Ich habe Hunger"]],
    ["Coding", ["Was bedeutet der % Operator in c++"]],
    ["Search", ["Wer ist der aktuelle Bundespräsident"]],
    [
        "Lokal",
        [
            "Plane einen 3tägigen Ausflug nach Mallorca",
            "Erstelle eine Liste mit dem tagesablauf für jeden Tag",
        ],
    ],
    [
        "Lokal",
        [
            "Classify if in that conversation jokes are made or not. Answer with Joke or No Joke"
        ],
    ],
]

for c in LOCALS_SEARCH_CONVERSATION:
    inputs = ""
    for ins in c[1]:
        inputs += USER_TOKEN + ins + END_TOKEN
    CLASSIFY_SEARCHABLE += "\n" + inputs + " --> " + c[0]
CLASSIFY_SEARCHABLE += "\n<|question|> -->"
