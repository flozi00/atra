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
]

for c in SEARCH_CONVERSATION:
    inputs = ""
    for ins in c[0]:
        inputs += USER_TOKEN + ins + END_TOKEN
    SEARCH_PROMPT += "\n" + inputs + " --> " + c[1]
SEARCH_PROMPT += "\n<|question|> -->"


QUERY_PROMPT = (
    "Formuliere Suchmaschinen Queries anhand der vorrausgehenden Konvesation:\n"
)
QUERY_CONVERSATION = [
    [["Was ist Chatgpt"], "Chatgpt Definition"],
    [["Ich habe Hunger"], "Einfache Rezepte"],
    [["Wer ist der aktuelle Bundespräsident"], "Aktueller Bundespräsident"],
    [["Sichere Programmierung"], "Sichere Programmierung"],
    [["Ich suche einen guten Artikel über .net Autorisierung"], ".net Autorisierung"],
    [["Wer ist Jeff Bezos"], "jeff Bezos"],
    [
        [
            "Wer ist Elon Musk",
            "Wieviele Kinder hat er",
            "Wer ist seine aktuelle Freundin",
        ],
        "Elon Musk aktuelle Freundin",
    ],
    [["Ich suche einen Artikel über Wallbox"], "Wallbox"],
    [["Wann iMac 2023"], "Wann ist das iMac 2023 Releasedatum"],
    [["Überwachungskamera"], "gute Überwachungskamera"],
    [["wann kommt gta 6 raus"], "GTA 6 Release"],
    [["Wer ist Angela Merkel", "Wann wurde sie geboren"], "Geburtsdatum Angela Merkel"],
    [["Wie ist das Wetter in Berlin", "und in München"], "Wetter in München"],
    [
        ["Wie ist das Wetter in Berlin", "und in München", "und in Hamburg"],
        "Wetter in Hamburg",
    ],
    [
        [
            "Was war die erste Partei von Angela Merkel ? ",
            "Seit wann ist sie Bundeskanzlerin ?",
        ],
        "Seit wann ist Angela Merkel Bundeskanzlerin",
    ],
    [
        [
            "Hi, wie geht es dir",
            "Was kannst du denn alles ?",
            "Schreibe einen Text über Primeline Solutions und warum man dort seine Server kaufen sollte",
            "Wie teuer ist denn eine H100",
        ],
        "Preis H100",
    ],
]

for c in QUERY_CONVERSATION:
    inputs = ""
    for ins in c[0]:
        inputs += USER_TOKEN + ins + END_TOKEN
    QUERY_PROMPT += "\n" + inputs + " --> " + c[1]
QUERY_PROMPT += "\n<|question|> -->"


CLASSIFY_SEARCHABLE = "Klassifiziere ob die Frage im Internet gesucht werden kann oder lokal beantwortet wird:\n"
LOCALS_SEARCH_CONVERSATION = [
    [["Wer bist du ?"], "Lokal"],
    [["Was kannst du ?"], "Lokal"],
    [["Und auf Deutsch ?"], "Lokal"],
    [["Plane einen 3tägigen Trip nach Hawaii"], "Lokal"],
    [["Wie ist das Wetter in Berlin", "und in München"], "Search"],
    [["Wie ist das Wetter in Berlin", "und in München", "und in Hamburg"], "Search"],
    [
        [
            "Was war die erste Partei von Angela Merkel ? ",
            "Seit wann ist sie Bundeskanzlerin ?",
        ],
        "Search",
    ],
    [["Was ist Chatgpt"], "Search"],
    [["Wer ist Stefan"], "Search"],
    [["Wer ist der geschäftsführer von Primeline"], "Search"],
    [["Ich habe Hunger"], "Lokal"],
    [["Wer ist der aktuelle Bundespräsident"], "Search"],
    [
        [
            "Plane einen 3tägigen Ausflug nach Mallorca",
            "Erstelle eine Liste mit dem tagesablauf für jeden Tag",
        ],
        "Lokal",
    ],
    [
        [
            "Classify if in that conversation jokes are made or not. Answer with Joke or No Joke"
        ],
        "Lokal",
    ],
]

for c in LOCALS_SEARCH_CONVERSATION:
    inputs = ""
    for ins in c[0]:
        inputs += USER_TOKEN + ins + END_TOKEN
    CLASSIFY_SEARCHABLE += "\n" + inputs + " --> " + c[1]
CLASSIFY_SEARCHABLE += "\n<|question|> -->"
