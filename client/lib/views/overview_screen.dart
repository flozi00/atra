import 'package:flutter/material.dart';

import 'package:highlight_text/highlight_text.dart';
import 'package:searchable_listview/searchable_listview.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../forms/subtitle.dart';
import '../utils/network.dart';
import './timeline_view.dart';
import 'package:woozy_search/woozy_search.dart';

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

  @override
  void initState() {
    super.initState();
    build_cards_list();
  }

  Widget cardItem(String value, String hash) {
    return Card(
        child: Column(children: [
      ListTile(
        title: Text(hash.substring(0, 10)),
        subtitle: Text(value.length > 50 ? value.substring(0, 50) : value),
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
                  onPressed: () {
                    build_timeline(hash, context, words).then((value) {
                      showDialog(
                          context: context,
                          builder: (BuildContext context) {
                            return Dialog(
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(20.0)),
                                child: value);
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
          ElevatedButton(
            onPressed: () async {
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
                                child: TextHighlight(
                              text: value,
                              words: words,
                              textStyle: TextStyle(
                                color:
                                    Theme.of(context).colorScheme.onBackground,
                              ),
                            ))));
                  });
            },
            child: const Text('Transkription'),
          ),
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
      SharedPreferences prefs = await SharedPreferences.getInstance();
      for (String mode in modes) {
        activeMode = mode;
        List<String> hashes = prefs.getStringList(mode) ?? [];
        for (String hash in hashes) {
          String firstElement = hash.split(",")[0];
          await get_transcription(firstElement).then((values) {
            cards.add(Transcription(hash: hash, transcription: values[0]));
            woozy.addEntry(values[0], value: hash);
            setState(() {});
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
          cardItem(trscp.transcription, trscp.hash),
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
        words = {};
        List<String> hashes = [];
        for (String word in q.split(" ")) {
          words[word] = HighlightedWord(
            textStyle: TextStyle(
              color: Theme.of(context).colorScheme.onPrimaryContainer,
              fontWeight: FontWeight.bold,
            ),
          );
          woozy.search(word).forEach((element) {
            if (element.score > 0.7) {
              hashes.add(element.value);
            }
          });
        }
        var result =
            list.where((element) => hashes.contains(element.hash)).toList();
        if (result.isEmpty) {
          return list;
        }
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

  Transcription({
    required this.hash,
    required this.transcription,
  });
}
