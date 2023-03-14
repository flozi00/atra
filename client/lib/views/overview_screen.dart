import 'package:flutter/material.dart';

import 'package:highlight_text/highlight_text.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../forms/asr.dart';
import '../utils/network.dart';
import './timeline_view.dart';
import 'package:woozy_search/woozy_search.dart';

import 'login.dart';

extension IterableExtension<T> on Iterable<T> {
  Iterable<T> distinctBy(Object Function(T e) getCompareValue) {
    var result = <T>[];
    forEach((element) {
      if (!result.any((x) => getCompareValue(x) == getCompareValue(element))) {
        result.add(element);
      }
    });

    return result;
  }
}

class OverviewScreen extends StatefulWidget {
  const OverviewScreen({Key? key}) : super(key: key);

  @override
  _OverviewScreenState createState() => _OverviewScreenState();
}

class _OverviewScreenState extends State<OverviewScreen> {
  List<Transcription> cards = [];
  List<Transcription> items = [];
  bool isFetching = false;
  Map<String, HighlightedWord> words = {};
  List<String> modes = ['asr'];
  Woozy<dynamic> woozy = Woozy();
  String question = "";
  String answer = "";
  Map<String, String> most_relevant = {};
  TextEditingController editingController = TextEditingController();

  @override
  void initState() {
    super.initState();
    build_cards_list();
    cards.sort((a, b) => b.score.compareTo(a.score));
  }

  Widget cardItem(String recognizedText, String hash, List<dynamic> listWords) {
    return Card(
        child: Column(children: [
      ListTile(
        title: Text(hash.substring(0, 20)),
        subtitle: Text(recognizedText.length > 250
            ? recognizedText.substring(0, 250).replaceAll("\n", " ")
            : recognizedText),
        onTap: () async {
          if (question.endsWith("?")) {
            await question_answering(question, most_relevant[hash]!)
                .then((result) {
              answer = result;
              for (String word in answer.split(" ")) {
                words[word] = HighlightedWord(
                    textStyle: TextStyle(
                  color: Theme.of(context).colorScheme.onPrimaryContainer,
                ));
              }
            });
          } else {
            answer = "";
          }
          build_timeline(hash, context, words, listWords).then((value) {
            showDialog(
                context: context,
                builder: (BuildContext context) {
                  return Dialog(
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20.0)),
                      child: Padding(
                          padding: const EdgeInsets.fromLTRB(15, 0, 15, 10),
                          child: Column(children: [
                            // if answer length > 2, display Text
                            if (answer.length >= 2)
                              Center(
                                  child: SizedBox(
                                      width: 360,
                                      child: Card(
                                        child: ListTile(
                                          title: Text(question),
                                          subtitle: Text(answer),
                                        ),
                                      ))),
                            const SizedBox(
                              height: 35,
                            ),
                            value
                          ])));
                });
          });
        },
      ),
      ButtonBar(
        children: <Widget>[
          ElevatedButton(
              onPressed: () async {
                SharedPreferences prefs = await SharedPreferences.getInstance();
                for (String mode in modes) {
                  List<String> hashList = prefs.getStringList(mode) ?? [];
                  hashList.remove(hash);
                  await prefs.setStringList(mode, hashList);
                }
                build_cards_list();
                Navigator.pushNamed(context, '/');
              },
              child: const Text('Delete'))
        ],
      ),
    ]));
  }

  Future<void> build_cards_list() async {
    if (isFetching == false) {
      isFetching = true;
      cards = [];
      String tokenValid = "Valid";
      SharedPreferences prefs = await SharedPreferences.getInstance();
      for (String mode in modes) {
        List<String> hashes = prefs.getStringList(mode) ?? [];
        for (String hash in hashes) {
          await get_transcription(hash).then((values) {
            cards.add(Transcription(
              hash: hash,
              transcription: values[0],
              words: values[1],
            ));
            tokenValid = values[2];
            for (int i = 0; i < values[1].length; i++) {
              String text = values[1][i]["text"];
              text = text.replaceAll("!", ".");
              text = text.replaceAll("?", ".");

              List<String> textList = text.split(".");

              for (String text in textList) {
                if (text != "") {
                  woozy.addEntry(text, value: hash);
                }
              }
            }
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
      items.addAll(cards);
    }
    isFetching = false;
  }

  void filterSearchResults(String query) {
    List<Transcription> dummyListData = [];
    words = {};
    most_relevant = {};

    if (query.isNotEmpty) {
      question = query.toLowerCase().trim();
      for (String word in question.split(" ")) {
        words[word] = HighlightedWord(
          textStyle: TextStyle(
            color: Theme.of(context).colorScheme.onPrimaryContainer,
          ),
        );
      }
      List<String> hashes = [];
      woozy.search(question).forEach((element) {
        if (element.score > 0.1) {
          hashes.add(element.value);
          var result = cards
              .where((elementResult) => hashes.contains(elementResult.hash))
              .toList();
          if (result.isNotEmpty) {
            for (int i = 0; i < result.length; i++) {
              if (result[i].score < element.score) {
                result[i].score += element.score;
              }
            }
          }

          if (most_relevant[element.value] == null) {
            most_relevant[element.value] = "";
          }
          most_relevant[element.value] =
              "${most_relevant[element.value]}\n${element.text}";
        }
      });

      var result = cards
          .where((elementResult) => hashes.contains(elementResult.hash))
          .toList();
      result = result.distinctBy((e) => e.hash).toList();
      dummyListData.addAll(result);

      setState(() {
        items.clear();
        items.addAll(dummyListData);
        items.sort((a, b) => b.score.compareTo(a.score));
      });
      return;
    } else {
      question = "";
      setState(() {
        items.clear();
        items.addAll(cards);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        floatingActionButton: FloatingActionButton(
          onPressed: () {
            showDialog(
                context: context,
                builder: (BuildContext context) {
                  return Dialog(
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20.0)),
                      child: const Center(
                          child: Padding(
                              padding: EdgeInsets.fromLTRB(25, 15, 25, 10),
                              child: ASRUpload())));
                });
          },
          child: const Icon(Icons.add_circle_outline),
        ),
        body: Column(
          children: [
            const SizedBox(
              height: 10,
            ),
            TextField(
              controller: editingController,
              onSubmitted: (value) {
                filterSearchResults(value);
              },
              onChanged: (value) {
                if (value.length <= 3) {
                  setState(() {
                    items.clear();
                    items.addAll(cards);
                  });
                }
              },
              decoration: const InputDecoration(
                  labelText: "Search",
                  hintText: "Search",
                  prefixIcon: Icon(Icons.search),
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.all(Radius.circular(25.0)))),
            ),
            const SizedBox(
              height: 25,
            ),
            SizedBox(
                height: MediaQuery.of(context).size.height - 200,
                child: SingleChildScrollView(
                    child: ListView.builder(
                        shrinkWrap: true,
                        itemCount: items.length,
                        itemBuilder: (context, i) {
                          return cardItem(items[i].transcription, items[i].hash,
                              items[i].words);
                        })))
          ],
        ));
  }
}

class Transcription {
  String hash;
  String transcription;
  List<dynamic> words;
  double score = 0;

  Transcription({
    required this.hash,
    required this.transcription,
    required this.words,
  });
}
