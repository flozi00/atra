import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import 'package:json_theme/json_theme.dart';

import './layout/bar.dart';
import 'utils/data.dart';
import 'views/overview_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final themeStr = await rootBundle.loadString('assets/appainter_theme.json');
  final themeJson = jsonDecode(themeStr);
  final theme = ThemeDecoder.decodeThemeData(themeJson)!;
  String? para1 = Uri.base.queryParameters["hash"];
  String? para2 = Uri.base.queryParameters["mode"];

  if (para1 != null && para2 != null) {
    await add_to_list(para1, para2);
  }
  runApp(MyApp(theme: theme));
}

// Add theme data to the MyApp constructor
class MyApp extends StatelessWidget {
  final ThemeData theme;

  const MyApp({
    Key? key,
    required this.theme,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // Pass the theme data to the MaterialApp
      theme: theme,
      debugShowCheckedModeBanner: false,
      initialRoute: '/',
      routes: {
        '/': (context) => const MyHomePage(),
      },
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({
    Key? key,
  }) : super(key: key);

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  void initState() {
    super.initState();
    //WidgetsBinding.instance.addPostFrameCallback((_) => check_login());
  }

  @override
  Widget build(BuildContext context) {
    return build_app_bar(
      context,
      [
        const Tab(icon: Icon(Icons.list), text: "Home"),
      ],
      [
        const OverviewScreen(),
        //const Assistant(),
      ],
    );
  }
}
