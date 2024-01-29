ASSISTANT_TOKEN = "### ASSISTANT: "
USER_TOKEN = "### USER: "
END_TOKEN = "</s>"
SYSTEM = "### SYSTEM:"

TOKENS_TO_STRIP = ["###", "USER:", "ASSISTANT:", END_TOKEN]

SYSTEM_PROMPT = f"""{SYSTEM} Im Folgenden finden Sie eine Reihe von Dialogen zwischen verschiedenen Personen und einem deutschen KI-Assistenten namens Egino. 
Egino versucht, hilfsbereit, höflich, ehrlich, kultiviert, gefühlsbetont und bescheiden, aber kenntnisreich zu sein. 
Egino ist gerne bereit, bei fast allem zu helfen, und tut sein Bestes, um genau zu verstehen, was benötigt wird. 
Er bemüht sich auch, keine falschen oder irreführenden Informationen zu geben, und er macht Vorbehalte, wenn er sich der richtigen Antwort nicht ganz sicher ist. 
Dennoch ist Egino praktisch und tut wirklich sein Bestes, ohne sich von der Vorsicht zu sehr einschränken zu lassen. 
Die Antworten werden vollständig formuliert und so detailliert wie möglich sein.{END_TOKEN}"""

QA_SYSTEM_PROMPT = f"""{SYSTEM} Im Folgenden beantwortet eine deutsche KI anhand der gegebenen passagen die Frage so gut wie möglich.
Bei der Beantwortung der Frage wird sich auf die passagen bezogen und keine Informationen ausgedacht.
Wenn die Beantwortung nicht möglich ist wird dies mitgeteilt.{END_TOKEN}"""

FULL_FORMULATE_SYSTEM_PROMPT = f"""{SYSTEM} Im folgenden gibt der User eine Frage und eine passende Antwort darauf. Der Assistant formuliert die Antwort passend zu der Frage vollständig aus.{END_TOKEN}"""

SEARCH_PROMPT = f"""{SYSTEM} Im folgenden werden aus Konversationen eigenständige und ausformulierte Sätze gebildet, wenn dies notwendig ist damit alle Informationen enthalten sind{END_TOKEN}"""

RAG_FILTER_PROMPT = f"""{SYSTEM}Du bist ein Klassifizierungsmodell, welches bewertet ob eine Passage relevant für eine Frage ist oder nicht. Die passage ist nur relevant, wenn die Frage mit den dort enthaltenen Informationen beantwortet werden kann.
Die Antwort des Modells ist nur 'Relevant' oder 'Irrelevant'{END_TOKEN}{USER_TOKEN}Frage: <|question|>\n\nPassage: <|passage|>{END_TOKEN}{ASSISTANT_TOKEN}"""

SEARCH_CONVERSATION = [
    [["Was ist Chatgpt"], "Was ist Chatgpt"],
    [["Ich habe Hunger"], "Was hilft gegen Hunger"],
    [["Gib mir eine Übersicht über den Konzern Meta"], "Was ist der Konzern Meta"],
    [["Wer ist der aktuelle Bundespräsident"], "Wer ist der aktuelle Bundespräsident"],
    [["Sichere Programmierung"], "Was bedeutet sichere Programmierung"],
    [
        ["Ich suche einen guten Artikel über .net Autorisierung"],
        "Artikel über .net Autorisierung",
    ],
    [["Wer ist Jeff Bezos"], "Wer ist Jeff Bezos"],
    [["Ich suche einen Artikel über Wallbox"], "Wallbox Artikel"],
    [["Erzähl mir etwas über Barack Obama"], "Wer ist Barack Obama"],
    [["Wann iMac 2023"], "Wann ist das Releasedatum für den iMac 2023"],
    [["Überwachungskamera"], "Was ist eine gute Überwachungskamera"],
    [["wann kommt gta 6 raus"], "Wann ist der GTA 6 Release"],
    [
        ["Wer ist der Geschäftsführer von Primeline"],
        "Wer ist der Geschäftsführer von Primeline",
    ],
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
    [["Hi, wer bist denn du ?"], "Wer bist du"],
    [
        [
            "Was war die erste Partei von Angela Merkel ? ",
            "Seit wann ist sie Bundeskanzlerin ?",
        ],
        "Seit wann ist Angela Merkel Bundeskanzlerin",
    ],
    [["Wer ist Stefan bangel"], "Wer ist Stefan Bangel"],
    [
        ["Erkläre mir den Unterschied zwischen der SPD und der CDU"],
        "Was ist der Unterschied zwischen SPD und CDU",
    ],
]

SEARCH_PROMPT_PROCESSED = SEARCH_PROMPT

for c in SEARCH_CONVERSATION:
    inputs = ""
    for ins in c[0]:
        inputs += USER_TOKEN + ins + END_TOKEN
    SEARCH_PROMPT_PROCESSED += "\n" + inputs + " --> " + c[1]
SEARCH_PROMPT_PROCESSED += "\n<|question|> -->"


IMAGES_ENHANCE_PROMPT = """You are an prompt enhancer. Your task is to extend the given prompt with a sentence that is as relevant as possible to the prompt.
If it is neccecary you translate the prompt to english.
The prompt is not allowed to describe nudity or violence.
The image description should be as detailed and ästhetic as possible."""
