import 'dart:async';

import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:shared_preferences/shared_preferences.dart';

GoogleSignIn _googleSignIn = GoogleSignIn(
  clientId:
      '118506021544-1bmtqohoa26r5e52169o4gp38pmf6g7m.apps.googleusercontent.com',
  scopes: <String>[
    'email',
  ],
);
GoogleSignInAccount? _currentUser;
String _bearerToken = '';

String bearerToken() => _bearerToken;
GoogleSignIn googleSignIn() => _googleSignIn;
GoogleSignInAccount getCurrentUser() => _currentUser!;
set currentUser(GoogleSignInAccount user) => _currentUser = user;
set BearerToken(String token) => _bearerToken = token;

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State createState() => LoginPageState();
}

class LoginPageState extends State<LoginPage> {
  final String _contactText = '';

  @override
  void initState() {
    super.initState();
    _googleSignIn.onCurrentUserChanged.listen((GoogleSignInAccount? account) {
      setState(() {
        _currentUser = account;
      });
      if (_currentUser != null) {
        _currentUser?.authHeaders.then((Map<String, String> headers) {
          _bearerToken = headers["Authorization"] ?? '';
          SharedPreferences.getInstance().then((prefs) {
            prefs.setString('token', _bearerToken);
          });
        });
      }
    });
  }

  Future<void> _handleSignIn() async {
    try {
      await _googleSignIn.signIn();
    } catch (error) {
      print(error);
    }
  }

  Future<void> _handleSignOut() => _googleSignIn.disconnect();

  Widget _buildBody() {
    final GoogleSignInAccount? user = _currentUser;
    if (user != null) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: <Widget>[
          ListTile(
            leading: GoogleUserCircleAvatar(
              identity: user,
            ),
            title: Text(user.displayName ?? ''),
            subtitle: Text(user.email),
          ),
          const Text('Signed in successfully.'),
          Text(_contactText),
          ElevatedButton(
            onPressed: _handleSignOut,
            child: const Text('SIGN OUT'),
          ),
        ],
      );
    } else {
      return Column(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: <Widget>[
          const Text('You are not currently signed in.'),
          ElevatedButton(
            onPressed: _handleSignIn,
            child: const Text('SIGN IN'),
          ),
        ],
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
        padding: const EdgeInsets.fromLTRB(50, 10, 50, 10),
        child: _buildBody());
  }
}
