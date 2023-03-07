import 'package:flutter/material.dart';

import 'package:highlight_text/highlight_text.dart';
import 'package:searchable_listview/searchable_listview.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../forms/subtitle.dart';
import '../utils/network.dart';
import './timeline_view.dart';
import 'package:woozy_search/woozy_search.dart';

import 'login.dart';

class OverviewScreen extends StatefulWidget {
  const OverviewScreen({Key? key}) : super(key: key);

  @override
  _OverviewScreenState createState() => _OverviewScreenState();
}

class _OverviewScreenState extends State<OverviewScreen> {
  List<Transcription> cards = [];
  bool isFetching = false;
  Map<String, HighlightedWord> words = {};
  List<String> modes = ['ocr', 'asr'];
  String activeMode = 'ocr';
  Woozy<dynamic> woozy = Woozy();
  String question = "";
  String answer = "";
  Map<String, String> most_relevant = {};

  @override
  void initState() {
    super.initState();
    build_cards_list();
  }

  void textDialog(BuildContext context, String recognizedText) {
    showDialog(
        context: context,
        builder: (BuildContext context) {
          return Dialog(
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20.0)),
              child: Padding(
                  padding: const EdgeInsets.only(
                      left: 20, top: 20, bottom: 20, right: 20),
                  child: SingleChildScrollView(
                      child: Column(children: [
                    // if answer length > 2, display Text
                    if (answer.length > 2)
                      Card(
                        child: ListTile(
                          title: Text(question),
                          subtitle: Text(answer),
                        ),
                      ),
                    TextHighlight(
                      text: recognizedText,
                      words: words,
                      textStyle: TextStyle(
                        color: Theme.of(context).colorScheme.onBackground,
                      ),
                    )
                  ]))));
        });
  }

  Widget cardItem(String recognizedText, String hash, List<dynamic> listWords) {
    return Card(
        child: Column(children: [
      ListTile(
        title: Text(hash.substring(0, 20)),
        subtitle: Text(recognizedText.length > 250
            ? recognizedText.substring(0, 250)
            : recognizedText),
      ),
      ButtonBar(
        children: <Widget>[
          /* The code does the following:
          1. Calls the build_timeline function and passes it the hash and context.
          2. The build_timeline function returns a Future with a Widget value.
          3. The Future is then checked for its value using "then".
          4. The Future's value is then used to generate a Dialog box.
          This is the code for the build_timeline function: */
          activeMode == "asr"
              ? ElevatedButton(
                  onPressed: () async {
                    if (question.endsWith("?")) {
                      await question_answering(question, most_relevant[hash]!)
                          .then((result) {
                        answer = result;
                        for (String word in answer.split(" ")) {
                          words[word] = HighlightedWord(
                              textStyle: TextStyle(
                            color: Theme.of(context)
                                .colorScheme
                                .onPrimaryContainer,
                          ));
                        }
                      });
                    }
                    build_timeline(hash, context, words, listWords)
                        .then((value) {
                      showDialog(
                          context: context,
                          builder: (BuildContext context) {
                            return Dialog(
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(20.0)),
                                child: Padding(
                                    padding: const EdgeInsets.fromLTRB(
                                        15, 0, 15, 10),
                                    child: Column(children: [
                                      // if answer length > 2, display Text
                                      if (answer.length > 2)
                                        Center(
                                            child: Card(
                                          child: ListTile(
                                            title: Text(question),
                                            subtitle: Text(answer),
                                          ),
                                        )),
                                      const SizedBox(
                                        height: 35,
                                      ),
                                      value
                                    ])));
                          });
                    });
                  },
                  child: const Text('Timeline'),
                )
              : const SizedBox(),
          /* The code does the following:
            1. Creates a dialog box with a Scrollview that displays a text.
            2. The text will be displayed in the dialog box. The text is taken from the variable value.
            3. The variable value is a string that contains the transcription of a speech. */
          activeMode != "asr"
              ? ElevatedButton(
                  onPressed: () {
                    if (question.endsWith("?")) {
                      question_answering(question, most_relevant[hash]!)
                          .then((result) {
                        answer = result;
                        for (String word in answer.split(" ")) {
                          words[word] = HighlightedWord(
                              textStyle: TextStyle(
                                  color: Theme.of(context)
                                      .colorScheme
                                      .onPrimaryContainer));
                        }
                        textDialog(context, recognizedText);
                      });
                    } else {
                      answer = "";
                      textDialog(context, recognizedText);
                    }
                  },
                  child: const Text('Transkription'),
                )
              : const SizedBox(),
          /* The code does the following:
          1. Creates an ElevatedButton that, when pressed, generates a Dialog that contains a file picker to select a video file
          2. Creates an ElevatedButton inside the dialog that, when pressed, uploads the video file to the server 
          and downloads the generated subtitled video file */
          activeMode == "asr"
              ? ElevatedButton(
                  onPressed: () async {
                    BuildContext dialogContext;
                    showDialog(
                        context: context,
                        builder: (BuildContext context) {
                          dialogContext = context;
                          return subtitle_form(context, hash);
                        });
                  },
                  child: const Text('Generate Video'),
                )
              : const SizedBox(),
          ElevatedButton(
              onPressed: () async {
                SharedPreferences prefs = await SharedPreferences.getInstance();
                for (String mode in modes) {
                  activeMode = mode;
                  List<String> hashList = prefs.getStringList(mode) ?? [];
                  hashList.remove(hash);
                  await prefs.setStringList(mode, hashList);
                }
                build_cards_list();
              },
              child: const Text('Delete'))
        ],
      ),
    ]));
  }

  /* The code does the following:
    1. Check if the data is already being fetched. If not, it fetches it from the local storage in the phone.
    2. It gets the first element of the hash (the image name).
    3. It gets the transcription of the image.
    4. It adds the transcription to the cards list.
    5. It sets the state.
    6. It repeats for all the hashes in the list. */
  Future<void> build_cards_list() async {
    if (isFetching == false) {
      isFetching = true;
      cards = [];
      String tokenValid = "Invalid";
      SharedPreferences prefs = await SharedPreferences.getInstance();
      for (String mode in modes) {
        activeMode = mode;
        List<String> hashes = prefs.getStringList(mode) ?? [];
        for (String hash in hashes) {
          await get_transcription(hash).then((values) {
            cards.add(Transcription(
                hash: hash, transcription: values[0], words: values[1]));
            tokenValid = values[2];
            for (int i = 0; i < values[1].length; i++) {
              woozy.addEntry(values[1][i]["text"], value: hash);
            }
            //woozy.addEntry(values[0], value: hash);
            setState(() {});
          });
        }
      }
      if (tokenValid != "Valid") {
        SharedPreferences prefs = await SharedPreferences.getInstance();
        if (prefs.getString('token') != null &&
            prefs.getString('token') != '') {
          showDialog(
              context: context,
              builder: (BuildContext context) {
                return const Dialog(child: LoginPage());
              });
        }
      }
    }
    isFetching = false;
  }

  @override
  Widget build(BuildContext context) {
    return SearchableList<Transcription>(
      builder: (Transcription trscp) =>
          cardItem(trscp.transcription, trscp.hash, trscp.words),
      initialList: cards,
      asyncListCallback: () async {
        await build_cards_list();
        while (isFetching == true) {
          await Future.delayed(const Duration(milliseconds: 10000));
        }
        print("asyncListCallback");

        return cards;
      },
      asyncListFilter: (q, list) {
        question = q.toLowerCase().trim();
        words = {};
        most_relevant = {};
        List<String> hashes = [];
        if (q.length < 5) return list;
        for (String word in q.split(" ")) {
          words[word] = HighlightedWord(
            textStyle: TextStyle(
              color: Theme.of(context).colorScheme.onPrimaryContainer,
            ),
          );
        }
        woozy.search(q).forEach((element) {
          hashes.add(element.value);
          if (most_relevant[element.value] == null) {
            most_relevant[element.value] = "";
          }
          most_relevant[element.value] =
              "${most_relevant[element.value]}\n${element.text}";
        });
        var result =
            list.where((element) => hashes.contains(element.hash)).toList();
        return result;
      },
      inputDecoration: InputDecoration(
        labelText: "Search",
        focusedBorder: OutlineInputBorder(
          borderSide: const BorderSide(
            width: 1.0,
          ),
          borderRadius: BorderRadius.circular(10.0),
        ),
      ),
    );
  }
}

class Transcription {
  String hash;
  String transcription;
  List<dynamic> words;

  Transcription({
    required this.hash,
    required this.transcription,
    required this.words,
  });
}
