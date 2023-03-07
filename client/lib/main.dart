import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import 'package:json_theme/json_theme.dart';

import './layout/bar.dart';
import 'views/login.dart';
import 'views/overview_screen.dart';
import 'views/upload.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final themeStr = await rootBundle.loadString('assets/appainter_theme.json');
  final themeJson = jsonDecode(themeStr);
  final theme = ThemeDecoder.decodeThemeData(themeJson)!;

  runApp(MyApp(theme: theme));
}

// Add theme data to the MyApp constructor
class MyApp extends StatelessWidget {
  final ThemeData theme;

  const MyApp({Key? key, required this.theme}) : super(key: key);

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
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  void initState() {
    super.initState();
    check_login();
    //WidgetsBinding.instance.addPostFrameCallback((_) => check_login());
  }

  void check_login() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    if (prefs.getString('token') == null || prefs.getString('token') == '') {
      showDialog(
          context: context,
          builder: (BuildContext context) {
            return const Dialog(child: LoginPage());
          });
    } else {
      BearerToken = prefs.getString('token')!;
    }
  }

  @override
  Widget build(BuildContext context) {
    return build_app_bar(
      context,
      [
        const Tab(icon: Icon(Icons.list), text: "Home"),
        const Tab(icon: Icon(Icons.upload_file), text: "Upload"),
      ],
      [
        const OverviewScreen(),
        const Upload(),
        //const Assistant(),
      ],
    );
  }
}
