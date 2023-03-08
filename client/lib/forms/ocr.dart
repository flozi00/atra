import 'dart:io';

import 'package:flutter/material.dart';

import 'package:flutter_form_builder/flutter_form_builder.dart';
import 'package:form_builder_file_picker/form_builder_file_picker.dart';
import 'package:form_builder_validators/form_builder_validators.dart';
import 'package:loader_overlay/loader_overlay.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../utils/data.dart';
import '../utils/network.dart';

class OCRUpload extends StatefulWidget {
  const OCRUpload({Key? key}) : super(key: key);

  @override
  _OCRUploadState createState() => _OCRUploadState();
}

class _OCRUploadState extends State<OCRUpload> {
  late GlobalKey<FormBuilderState> uploadFormKey;
  List<String> languages = [
    'Handwritten',
    'Printed',
  ];

  @override
  void initState() {
    super.initState();

    uploadFormKey = GlobalKey<FormBuilderState>();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
        width: 420,
        child: LoaderOverlay(
            child: FormBuilder(
          key: uploadFormKey,
          child: Column(children: <Widget>[
            const SizedBox(
              height: 50,
            ),
            // 1. Create a form builder instance
            FormBuilderDropdown<String>(
              // 2. Give the form builder a name
              name: 'mode',

              // 3. Add decoration to the form builder
              decoration: const InputDecoration(
                labelText: 'Form of text',
                helperText: 'Select which type of text you want to recognize',
              ),

              // 4. Add a validator to the form builder
              validator: FormBuilderValidators.compose(
                [FormBuilderValidators.required()],
              ),

              // 5. Create a list of items for the form builder
              items: languages
                  .map((lang) => DropdownMenuItem(
                        alignment: AlignmentDirectional.center,
                        value: lang,
                        child: Text(lang),
                      ))
                  .toList(),

              // 6. Add a value transformer to the form builder
              valueTransformer: (val) => val?.toString(),
            ),
            const SizedBox(
              height: 5,
            ),
            // 1. Add the FormBuilderFilePicker widget to the form
            FormBuilderFilePicker(
              name: 'media',
              maxFiles: 1,
              previewImages: false,
              decoration: const InputDecoration(
                labelText: 'Image',
              ),
              validator: FormBuilderValidators.compose(
                [FormBuilderValidators.required()],
              ),
              allowedExtensions: const ['png', 'jpg', 'JPEG', "pdf"],
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
              height: 5,
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
                  var audio =
                      uploadFormKey.currentState!.value["media"][0].bytes;
                  if (audio == null) {
                    await File(
                            uploadFormKey.currentState!.value["media"][0].path)
                        .readAsBytes()
                        .then((value) => audio = value);
                  }
                  String audioName =
                      uploadFormKey.currentState!.value["media"][0].name;
                  var base64Audio = uint8ListTob64(audio);

                  SendToOCR(
                    base64Audio,
                    audioName,
                    uploadFormKey.currentState!.value["srclang"]
                        .toString()
                        .toLowerCase(),
                  ).then((String newhash) async {
                    SharedPreferences prefs =
                        await SharedPreferences.getInstance();
                    List<String> hashes = prefs.getStringList("ocr") ?? [];
                    if (hashes.contains(newhash) == false) {
                      hashes.add(newhash);
                    }
                    prefs.setStringList("ocr", hashes);
                    context.loaderOverlay.hide();
                    Navigator.pushReplacementNamed(context, "/");
                  });
                }
              },
              child: const Text('Recognize Text'),
            ),
          ]),
        )));
  }
}
