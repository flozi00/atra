import 'dart:convert';
import 'dart:typed_data';

// The function uint8ListTob64 converts a Uint8List into a base64 encoded string.
String uint8ListTob64(Uint8List uint8list) {
  String base64String = base64Encode(uint8list);
  return base64String;
}

// This function converts a base64 encoded string to a list of bytes
Uint8List b64Touint8List(String b64) {
  Uint8List bytes = base64Decode(b64);
  return bytes;
}

//This function is used to format time in second to minute:second
String formatedTime({required int timeInSecond}) {
  //get the second
  int sec = timeInSecond % 60;
  //get the minute
  int min = (timeInSecond / 60).floor();
  //check if minute is less than 10, if so add 0 in front
  String minute = min.toString().length <= 1 ? "0$min" : "$min";
  //check if second is less than 10, if so add 0 in front
  String second = sec.toString().length <= 1 ? "0$sec" : "$sec";
  //return formated time
  return "$minute : $second";
}
