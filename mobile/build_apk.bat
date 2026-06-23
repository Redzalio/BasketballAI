@echo off
REM ============================================================
REM  HoopTracker mobile - build the offline debug APK.
REM  Run this after changing anything in www\ (e.g. swapping the
REM  model in www\models\detector.onnx). Output APK path is printed
REM  at the end. Needs internet only the FIRST time (Gradle deps).
REM ============================================================
set "JAVA_HOME=C:\Users\USER\Tools\jdk21\jdk-21.0.11+10"
set "ANDROID_HOME=C:\Users\USER\Tools\android-sdk"
set "ANDROID_SDK_ROOT=C:\Users\USER\Tools\android-sdk"

cd /d "%~dp0"
echo === Syncing web assets into the Android project ===
call npx cap sync android || goto :err

echo === Building debug APK ===
cd android
REM 'clean' is intentional: after swapping the model, incremental Gradle can stay
REM UP-TO-DATE and ship a stale APK with the old model. clean forces a full repackage.
call gradlew.bat clean assembleDebug || goto :err

echo.
echo ============================================================
echo  DONE. APK:
echo  %~dp0android\app\build\outputs\apk\debug\app-debug.apk
echo ============================================================
pause
exit /b 0

:err
echo.
echo !!! BUILD FAILED - see the error above.
pause
exit /b 1
