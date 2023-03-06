import 'dart:convert';

import 'package:flutter/foundation.dart';

import 'package:http/http.dart' as http;

final headers = {
  'authority': 'asr.a-ware.io',
  'accept': '*/*',
  'accept-language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
  'content-type': 'application/json',
  'origin': 'https://asr.a-ware.io',
  'Accept-Encoding': 'gzip',
};

String myurl = Uri.base.origin.toString();

String baseURL = myurl.contains("localhost")
    ? "http://127.0.0.1:7860/gradio"
    : "$myurl/gradio";

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
      headers: headers, body: jsonEncode(params));
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // get the hash of the translation result
  String newhash = jsonDecode(utf8.decode(res.bodyBytes))["data"][0];
  return newhash;
}

Future<String> SendToOCR(var image, String name, String model) async {
  var params = {
    "data": [
      "data:@file/octet-stream;base64,$image",
      "large",
      model,
    ]
  };

  // send audio data to the translation task
  var res = await http.post(Uri.parse('$baseURL/run/ocr'),
      headers: headers, body: jsonEncode(params));
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // get the hash of the translation result
  String newhash = jsonDecode(utf8.decode(res.bodyBytes))["data"][0];
  return newhash;
}

// This code fetches the transcription from the database for a given hash.
Future<List<dynamic>> get_transcription(String hash) async {
  // Define the parameters to be sent to the server.
  var params = {
    "data": [hash]
  };

  // Send a POST request to the server with the parameters.
  var res = await http.post(Uri.parse('$baseURL/run/get_transcription'),
      headers: headers, body: jsonEncode(params));
  // Check that the response from the server is valid.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // Parse the response from the server and return the result.
  List<dynamic> timestamps = jsonDecode(utf8.decode(res.bodyBytes))["data"];
  return timestamps;
}

// This function takes a hash and a video file and returns a byte array
// of the video file.
Future<Uint8List> get_video(String hash, String mediafile) async {
  // This is the body of the request that is sent to the server.
  var params = {
    "data": [
      hash,
      {"name": "media.mp4", "data": "data:@file/octet-stream;base64,$mediafile"}
    ]
  };

  // Send the request to the server.
  var res = await http.post(Uri.parse('$baseURL/run/subtitle'),
      headers: headers, body: jsonEncode(params));

  // Check that the request was successful.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // Get the name of the file that the server sent back.
  String video =
      Uri.encodeFull(jsonDecode(utf8.decode(res.bodyBytes))["data"][0]["name"]);

  // Get the file from the server.
  res = await http.get(Uri.parse('$baseURL/file=$video'));

  // Convert the file to a byte array.
  Uint8List myvideo = res.bodyBytes;

  // Return the byte array.
  return myvideo;
}

// This function takes a hash and a video file and returns a byte array
// of the video file.
Future<String> get_audio(String hash) async {
  // This is the body of the request that is sent to the server.
  var params = {
    "data": [
      hash,
    ]
  };

  // Send the request to the server.
  var res = await http.post(Uri.parse('$baseURL/run/get_audio'),
      headers: headers, body: jsonEncode(params));

  // Check that the request was successful.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }

  // Get the name of the file that the server sent back.
  String video =
      Uri.encodeFull(jsonDecode(utf8.decode(res.bodyBytes))["data"][0]["name"]);

  // Return the byte array.
  return Uri.parse('$baseURL/file=$video').toString();
}

Future<bool> update_transcript(String hash, String text) async {
  var params = {
    "data": [
      hash,
      text,
    ]
  };

  // Send the request to the server.
  var res = await http.post(Uri.parse('$baseURL/run/correct_transcription'),
      headers: headers, body: jsonEncode(params));

  // Check that the request was successful.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }
  return true;
}

Future<String> question_answering(String question, String context) async {
  var headers = {
    //'Authorization': 'Bearer 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',
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
      headers: headers, body: jsonEncode(params));

  // Check that the request was successful.
  if (res.statusCode != 200) {
    throw Exception('http.post error: statusCode= ${res.statusCode}');
  }
}
