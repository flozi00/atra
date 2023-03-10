import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'package:flutter_form_builder/flutter_form_builder.dart';
import 'package:form_builder_file_picker/form_builder_file_picker.dart';
import 'package:form_builder_validators/form_builder_validators.dart';
import 'package:loader_overlay/loader_overlay.dart';

import '../utils/data.dart';
import '../utils/network.dart';
import 'package:microphone/microphone.dart';
import 'package:siri_wave/siri_wave.dart';

class ASRUpload extends StatefulWidget {
  const ASRUpload({Key? key}) : super(key: key);

  @override
  _ASRUploadState createState() => _ASRUploadState();
}

class _ASRUploadState extends State<ASRUpload> {
  late GlobalKey<FormBuilderState> uploadFormKey;
  late MicrophoneRecorder microphoneRecorder;
  Uint8List audioBytes = Uint8List(0);
  bool isRecording = false;
  List<String> languages = [
    'English',
    'French',
    'German',
    'Italian',
    'Russian',
    'Spanish'
  ];

  @override
  void initState() {
    super.initState();
    microphoneRecorder = MicrophoneRecorder()..init();

    uploadFormKey = GlobalKey<FormBuilderState>();
  }

  Future<void> uploadASR(Uint8List audio, String lang, String audioName) async {
    var base64Audio = uint8ListTob64(audio);

    await SendToASR(base64Audio, audioName, lang, "large")
        .then((String newhash) async {
      await add_to_list(newhash, "asr");
    });
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
        child: LoaderOverlay(
            child: FormBuilder(
      key: uploadFormKey,
      child: Column(children: <Widget>[
        const SizedBox(
          height: 50,
        ),
        // Add the FormBuilderDropdown to the form
        FormBuilderDropdown<String>(
          name: 'srclang',
          decoration: const InputDecoration(
            labelText: 'Target language',
            helperText:
                'Select the target language to translate the Audio / Video file',
          ),

          // Add the validator to ensure that the user has selected a language
          validator:
              FormBuilderValidators.compose([FormBuilderValidators.required()]),

          // Add the list of languages that will be displayed in the drop down
          items: languages
              .map((lang) => DropdownMenuItem(
                    alignment: AlignmentDirectional.center,
                    value: lang,
                    child: Text(lang),
                  ))
              .toList(),

          // Transform the value to a string to be used in the MediaInfo call
          valueTransformer: (val) => val?.toString(),
        ),
        const SizedBox(
          height: 5,
        ),
        // 1. Add the FormBuilderFilePicker widget to the form
        FormBuilderFilePicker(
          name: 'media',
          maxFiles: 10,
          allowMultiple: true,
          previewImages: false,
          decoration: const InputDecoration(
            labelText: 'Audio / Video',
          ),
          // list of allowed file extensions for audio files and video files
          allowedExtensions: const [
            'mp3',
            'wav',
            "m4a",
            "mp4",
            "mov",
            "avi",
            "mkv",
            "flv",
            "wmv",
            "webm",
            "ogg",
            "3gp",
            "3g2",
            "mpeg",
            "mpg",
            "m4v",
            "f4v",
            "f4p",
            "f4a",
            "f4b"
          ],
          typeSelectors: [
            // 2. Add a TypeSelector to the list of typeSelectors
            TypeSelector(
              // 3. Set the type to FileType.custom
              type: FileType.custom,
              // 4. Set the selector to a widget of your choice
              selector: Row(
                children: const <Widget>[
                  Icon(Icons.file_upload),
                  Text('Upload'),
                ],
              ),
            )
          ],
        ),
        const SizedBox(
          height: 25,
        ),
        IconButton(
            iconSize: 45,
            onPressed: () async {
              if (isRecording) {
                isRecording = false;
                await microphoneRecorder.stop();
                await microphoneRecorder.toBytes().then((value) {
                  setState(() {
                    audioBytes = value;
                  });
                });
                microphoneRecorder.dispose();
                microphoneRecorder = MicrophoneRecorder();
                await microphoneRecorder.init();
              } else {
                isRecording = true;
                await microphoneRecorder.start();
                setState(() {});
              }
            },
            icon: isRecording == false
                ? const Icon(Icons.mic_off)
                : const Icon(Icons.mic)),
        const SizedBox(height: 10),
        isRecording ? SiriWave() : const Text("Click to start recording"),
        const SizedBox(
          height: 25,
        ),
        ElevatedButton(
          /* The code does the following:
                1. Makes a call to a function that will upload the audio file to the server.
                2. Retrieves the hash returned by the server.
                3. Adds the hash to a list of hashes in shared preferences. This is done to make sure that the user will not be able to upload the same audio file twice.
                4. Hides the loader overlay.
                5. Pushes "/" to the Navigator, which will take the user back to the home page. */
          onPressed: () async {
            if (uploadFormKey.currentState!.validate()) {
              uploadFormKey.currentState?.save();
              context.loaderOverlay.show();
              if (uploadFormKey.currentState!.value["media"] != null) {
                for (int i = 0;
                    i < uploadFormKey.currentState!.value["media"].length;
                    i++) {
                  var audio =
                      uploadFormKey.currentState!.value["media"][i].bytes;
                  if (audio == null) {
                    await File(
                            uploadFormKey.currentState!.value["media"][i].path)
                        .readAsBytes()
                        .then((value) => audio = value);
                  }
                  String audioName =
                      uploadFormKey.currentState!.value["media"][i].name;
                  String lang = uploadFormKey.currentState!.value["srclang"]
                      .toString()
                      .toLowerCase();
                  await uploadASR(audio, lang, audioName);
                }
              }
              if (audioBytes.length > 10) {
                String audioName = "microphone.wav";
                String lang = uploadFormKey.currentState!.value["srclang"]
                    .toString()
                    .toLowerCase();
                await uploadASR(audioBytes, lang, audioName);
              }
              context.loaderOverlay.hide();
              Navigator.pushReplacementNamed(context, "/");
            }
          },
          child: const Text('Transcribe Audio'),
        ),
      ]),
    )));
  }
}
