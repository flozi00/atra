import 'package:atra/utils/network.dart';
import 'package:flutter/material.dart';

import 'package:flutter_form_builder/flutter_form_builder.dart';
import 'package:loader_overlay/loader_overlay.dart';

class Assistant extends StatefulWidget {
  const Assistant({Key? key}) : super(key: key);

  @override
  _AssistantState createState() => _AssistantState();
}

class _AssistantState extends State<Assistant> {
  late GlobalKey<FormBuilderState> uploadFormKey;
  // ignore: prefer_final_fields
  List<Widget> _searchResults = [];

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
            FormBuilderTextField(
                name: "search",
                decoration: const InputDecoration(labelText: "Search Query")),
            const SizedBox(
              height: 6,
            ),
            ElevatedButton(
              onPressed: () async {
                if (uploadFormKey.currentState!.validate()) {
                  uploadFormKey.currentState?.save();
                  context.loaderOverlay.show();
                  search_web(uploadFormKey.currentState!.value['search'])
                      .then((value) {
                    setState(() {
                      for (int index = 0; index < value.length; index++) {
                        _searchResults.add(Text(value[index]["title"]));
                      }
                    });
                    context.loaderOverlay.hide();
                  });
                }
              },
              child: const Text('Search'),
            ),
          ]),
        )));
  }
}
