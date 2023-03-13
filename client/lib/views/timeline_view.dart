import 'package:flutter/material.dart';

import 'package:highlight_text/highlight_text.dart';
import 'package:timeline_tile/timeline_tile.dart';

import '../utils/data.dart';
import '../utils/network.dart';
import 'package:just_audio/just_audio.dart';

Future<Widget> build_timeline(String args, BuildContext context,
    Map<String, HighlightedWord> words, List<dynamic> values) async {
  List<Step> details = [];
  _TimelineActivity widget = const _TimelineActivity(
    steps: [],
    words: {},
  );
  for (int count = 0; count < values.length; count++) {
    var value = values[count];
    int startTime = value['start_timestamp'];
    int stopTime = value["stop_timestamp"];
    Step tile = Step(
      type: Type.checkpoint,
      hour: formatedTime(timeInSecond: startTime),
      message: value["text"],
      duration: 0,
      color: Theme.of(context).colorScheme.onPrimary,
      icon: Icons.mic_none,
      hash: value["id"],
      full_hash: args,
    );
    Step bind = Step(
      type: Type.line,
      hour: "",
      message: "",
      duration: stopTime - startTime,
      color: Theme.of(context).colorScheme.onSecondary,
      icon: Icons.mic_none,
      hash: value["id"],
      full_hash: args,
    );
    details.add(tile);
    details.add(bind);
    if (count + 1 < values.length) {
      details.add(Step(
        type: Type.line,
        hour: "",
        message: "",
        duration: values[count + 1]['start_timestamp'] - stopTime,
        color: Theme.of(context).colorScheme.onSecondary,
        icon: Icons.mic_none,
        hash: value["id"],
        full_hash: args,
      ));
    }
  }
  widget = _TimelineActivity(steps: details, words: words);

  return SizedBox(
      height: MediaQuery.of(context).size.height - 250, child: widget);
}

class _TimelineActivity extends StatelessWidget {
  const _TimelineActivity({Key? key, required this.steps, required this.words})
      : super(key: key);

  final List<Step> steps;
  final Map<String, HighlightedWord> words;

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      shrinkWrap: true,
      physics: const AlwaysScrollableScrollPhysics(),
      itemCount: steps.length,
      itemBuilder: (BuildContext context, int index) {
        final Step step = steps[index];

        final IndicatorStyle indicator = step.isCheckpoint
            ? _indicatorStyleCheckpoint(step)
            : const IndicatorStyle(width: 0);

        final righChild = _RightChildTimeline(step: step, words: words);

        Widget leftChild = _LeftChildTimeline(step: step);

        return TimelineTile(
          alignment: TimelineAlign.manual,
          isFirst: index == 0,
          isLast: index == steps.length - 1,
          lineXY: 0.25,
          indicatorStyle: indicator,
          startChild: leftChild,
          endChild: righChild,
          hasIndicator: step.isCheckpoint,
          beforeLineStyle: LineStyle(
            color: step.color,
            thickness: 8,
          ),
        );
      },
    );
  }

  IndicatorStyle _indicatorStyleCheckpoint(Step step) {
    return IndicatorStyle(
      width: 46,
      height: 64,
      indicator: Container(
        decoration: BoxDecoration(
          color: step.color,
          borderRadius: const BorderRadius.all(
            Radius.circular(32),
          ),
        ),
        child: Center(
            child: InkWell(
          child: Icon(
            step.icon,
            size: 32,
            color: Colors.white,
          ),
          onTap: () async {
            final player = AudioPlayer(); // Create a player
            await get_audio(step.hash).then((value) async {
              await player.setUrl(value);
              await player.play();
            });
          },
        )),
      ),
    );
  }
}

class _RightChildTimeline extends StatelessWidget {
  const _RightChildTimeline({Key? key, required this.step, required this.words})
      : super(key: key);

  final Step step;
  final Map<String, HighlightedWord> words;

  @override
  Widget build(BuildContext context) {
    final double minHeight =
        step.isCheckpoint ? 1 : step.duration.toDouble() + 5 * 10;

    return ConstrainedBox(
      constraints: BoxConstraints(minHeight: minHeight),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          Padding(
              padding:
                  const EdgeInsets.only(left: 20, top: 8, bottom: 8, right: 8),
              child: InkWell(
                child: TextHighlight(
                  text: step.message,
                  words: words,
                  textStyle: TextStyle(
                    color: Theme.of(context).colorScheme.onBackground,
                  ),
                ),
                onTap: () {
                  showDialog(
                      context: context,
                      builder: (BuildContext context) {
                        TextEditingController controller =
                            TextEditingController(text: step.message);
                        return Dialog(
                            child: Padding(
                                padding: const EdgeInsets.all(25.0),
                                child: Column(
                                  children: [
                                    const Text("Edit Transcript"),
                                    const SizedBox(
                                      height: 10,
                                    ),
                                    TextFormField(
                                      controller: controller,
                                      maxLines: 20,
                                      minLines: 3,
                                      decoration: const InputDecoration(
                                          border: OutlineInputBorder()),
                                    ),
                                    const SizedBox(
                                      height: 10,
                                    ),
                                    ElevatedButton(
                                        onPressed: () async {
                                          await update_transcript(step.hash,
                                              controller.text, step.full_hash);
                                          step.message = controller.text;
                                          Navigator.pushReplacementNamed(
                                              context, "/");
                                        },
                                        child: const Text("Save"))
                                  ],
                                )));
                      });
                },
              )),
          if (step.isCheckpoint)
            Row(
              children: [
                const SizedBox(
                  width: 15,
                ),
                ElevatedButton.icon(
                    onPressed: () async {
                      await do_voting(step.hash, "bad");
                      Navigator.pushReplacementNamed(context, "/");
                    },
                    icon: const Icon(Icons.repeat),
                    label: const Text("Regenerate")),
              ],
            )
        ],
      ),
    );
  }
}

class _LeftChildTimeline extends StatelessWidget {
  const _LeftChildTimeline({Key? key, required this.step}) : super(key: key);

  final Step step;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: <Widget>[
        Padding(
          padding: const EdgeInsets.only(right: 10, top: 32),
          child: Text(step.hour,
              textAlign: TextAlign.center,
              style: TextStyle(color: Theme.of(context).colorScheme.primary)),
        )
      ],
    );
  }
}

enum Type {
  checkpoint,
  line,
}

class Step {
  Step({
    required this.type,
    required this.hour,
    required this.message,
    required this.duration,
    required this.color,
    required this.icon,
    required this.hash,
    required this.full_hash,
  });

  final Type type;
  final String hour;
  String message;
  final int duration;
  final Color color;
  final IconData icon;
  final String hash;
  final String full_hash;

  bool get isCheckpoint => type == Type.checkpoint;

  bool get hasHour => hour.isNotEmpty;
}
