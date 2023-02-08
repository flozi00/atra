import 'package:flutter/material.dart';

import 'package:file_saver/file_saver.dart';
import 'package:flutter_form_builder/flutter_form_builder.dart';
import 'package:form_builder_file_picker/form_builder_file_picker.dart';
import 'package:loader_overlay/loader_overlay.dart';

import '../utils/data.dart';
import '../utils/network.dart';

Dialog subtitle_form(BuildContext dialogContext, String hash) {
  GlobalKey<FormBuilderState> burninForm = GlobalKey<FormBuilderState>();
  return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.0)),
      child: LoaderOverlay(
          child: SingleChildScrollView(
              child: FormBuilder(
        key: burninForm,
        child: Column(children: [
          const SizedBox(
            height: 50,
          ),
          FormBuilderFilePicker(
            name: "media",
            maxFiles: 1,
            previewImages: false,
            decoration: const InputDecoration(
              labelText: 'Audio / Video',
            ),
            allowedExtensions: const [
              "mp4",
            ],
            typeSelectors: [
              TypeSelector(
                type: FileType.custom,
                selector: Row(
                  children: const <Widget>[
                    Icon(Icons.file_upload),
                    Text('Upload'),
                  ],
                ),
              )
            ],
          ),
          ElevatedButton(
            onPressed: () async {
              if (burninForm.currentState!.validate()) {
                burninForm.currentState?.save();
                dialogContext.loaderOverlay.show();
                var audio = burninForm.currentState!.value["media"][0].bytes;

                var mediafile = uint8ListTob64(audio);

                await get_video(hash, mediafile).then((bytes) async {
                  await FileSaver.instance
                      .saveFile("subtitled_video.mp4", bytes, "mp4");
                  dialogContext.loaderOverlay.hide();
                  Navigator.pop(dialogContext);
                });
              }
            },
            child: const Text('Generate subtitled video'),
          ),
        ]),
      ))));
}
