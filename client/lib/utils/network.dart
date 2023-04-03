import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../views/login.dart' as login;
import 'package:hive/hive.dart';
import 'package:crypto/crypto.dart';

Map<String, String> headers() {
  return {
    'accept': '*/*',
    'accept-language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
    'content-type': 'application/json',
    'Accept-Encoding': 'gzip',
    "Authorization": login.bearerToken(),
  };
}

String myurl = Uri.base.origin.toString();

String baseURL = myurl.contains("localhost")
    ? "http://127.0.0.1:7860/gradio"
    : myurl.contains("github.io")
        ? "https://atra.ai/gradio"
        : "$myurl/gradio";

var transcription_box;

//The function sendtotask sends the audio file to the server and returns the hash string of the task.
Future<String> SendToASR(
    var audio, String name, String source, String model) async {
  var params = {
    "data": [
      {"name": name, "data": "data:@file/octet-stream;base64,$audio"},
      source,
      model
    ]
  };

  // send audio data to the translation task
  var res = await http.post(Uri.parse('$baseURL/run/transcription'),
      headers: headers(), body: jsonEncode(params));
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // get the hash of the translation result
  String newhash = jsonDecode(utf8.decode(res.bodyBytes))["data"][0];
  return newhash;
}

// This code fetches the transcription from the database for a given hash.
Future<List<dynamic>> get_transcription(String hash) async {
  var params = {
    "data": [hash]
  };

  String compressedHash = sha256.convert(utf8.encode(hash)).toString();
  transcription_box ??= await Hive.openBox("transcription");
  // Check if the transcription is already in the local storage.
  var timestamps = transcription_box.get(compressedHash, defaultValue: null);
  if (timestamps != null) {
    return timestamps;
  }

  var res = await http.post(Uri.parse('$baseURL/run/get_transcription'),
      headers: headers(), body: jsonEncode(params));
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  timestamps = jsonDecode(utf8.decode(res.bodyBytes))["data"];
  if (timestamps[0].contains("***") == false) {
    transcription_box.put(compressedHash, timestamps);
  }
  return timestamps;
}

// This function takes a hash and a video file and returns a byte array
// of the video file.
Future<String> get_audio(String hash) async {
  var params = {
    "data": [
      hash,
    ]
  };

  var res = await http.post(Uri.parse('$baseURL/run/get_audio'),
      headers: headers(), body: jsonEncode(params));

  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  String video =
      Uri.encodeFull(jsonDecode(utf8.decode(res.bodyBytes))["data"][0]["name"]);

  return Uri.parse('$baseURL/file=$video').toString();
}

Future<bool> update_transcript(
    String hash, String text, String fullHash) async {
  var params = {
    "data": [
      hash,
      text,
    ]
  };

  // Send the request to the server.
  var res = await http.post(Uri.parse('$baseURL/run/correct_transcription'),
      headers: headers(), body: jsonEncode(params));

  // Check that the request was successful.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  String compressedHash = sha256.convert(utf8.encode(fullHash)).toString();
  transcription_box ??= await Hive.openBox("transcription");
  transcription_box.put(compressedHash, null);

  return true;
}

Future<String> question_answering(String question, String context) async {
  var headers = {
    'Content-Type': 'application/json',
  };

  var data = jsonEncode({"inputs": "$context \n $question"});

  var url =
      Uri.parse('https://api-inference.huggingface.co/models/google/flan-ul2');
  var res = await http.post(url, headers: headers, body: data);
  if (res.statusCode != 200) {
    print(res.body);
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // Get the name of the file that the server sent back.
  String answer = jsonDecode(utf8.decode(res.bodyBytes))[0]["generated_text"];

  // Return the byte array.
  return answer;
}

Future<void> do_voting(String hash, String text) async {
  var params = {
    "data": [
      hash,
      text,
    ]
  };

  // Send the request to the server.
  var res = await http.post(Uri.parse('$baseURL/run/vote_result'),
      headers: headers(), body: jsonEncode(params));

  // Check that the request was successful.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }
}

Future<String> get_video_subs(String hash) async {
  // This is the body of the request that is sent to the server.
  var params = {
    "data": [
      hash,
    ]
  };

  // Send the request to the server.
  var res = await http.post(Uri.parse('$baseURL/run/subtitle'),
      headers: headers(), body: jsonEncode(params));

  // Check that the request was successful.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // Get the name of the file that the server sent back.
  String subs =
      Uri.encodeFull(jsonDecode(utf8.decode(res.bodyBytes))["data"][0]["name"]);

  // Get the file from the server.
  var sub = await http.get(Uri.parse('$baseURL/file=$subs'));

  // Convert the file to a byte array.
  Uint8List subBytes = sub.bodyBytes;

  // Return the byte array.
  return utf8.decode(subBytes);
}
