import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ecowhisky_atc/main.dart';

void main() {
  testWidgets('renders PTT button and feedback', (WidgetTester tester) async {
    await tester.pumpWidget(const EcoWhiskyApp());
    expect(find.byIcon(Icons.mic), findsOneWidget);
    expect(find.text('Feedback:'), findsOneWidget);
  });
}
