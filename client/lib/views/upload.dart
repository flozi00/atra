import 'package:atra/forms/asr.dart';
import 'package:atra/forms/ocr.dart';
import 'package:flutter/material.dart';

class Upload extends StatefulWidget {
  const Upload({Key? key}) : super(key: key);

  @override
  _UploadState createState() => _UploadState();
}

class _UploadState extends State<Upload> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Card(
          child: Column(children: [
            const ListTile(
              title: Text("Speech"),
              subtitle: Text(
                  "Upload a media file containing speech you want to transcribe."),
            ),
            ButtonBar(
              children: [
                ElevatedButton(
                  onPressed: () {
                    showDialog(
                        context: context,
                        builder: (BuildContext context) {
                          return Dialog(
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20.0)),
                              child: const Center(child: ASRUpload()));
                        });
                  },
                  child: const Text("Upload"),
                ),
              ],
            ),
          ]),
        ),
        Card(
          child: Column(children: [
            const ListTile(
              title: Text("Text"),
              subtitle: Text(
                  "Upload a media file containing text you want to recognize."),
            ),
            ButtonBar(
              children: [
                ElevatedButton(
                  onPressed: () {
                    showDialog(
                        context: context,
                        builder: (BuildContext context) {
                          return Dialog(
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20.0)),
                              child: const Center(child: OCRUpload()));
                        });
                  },
                  child: const Text("Upload"),
                ),
              ],
            ),
          ]),
        )
      ],
    );
  }
}
