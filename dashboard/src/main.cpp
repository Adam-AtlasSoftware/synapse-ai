#include <QGuiApplication>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QTimer>
#include <QQmlContext>

#include "EngineBridge.h"

// The dashboard entry point. It creates the engine bridge (which builds a default
// model on the GPU), exposes it to QML as the `bridge` context property, and hands
// off to Main.qml. Everything the UI shows flows through that one object.
int main(int argc, char* argv[]) {
  QGuiApplication app(argc, argv);
  // Gives QSettings a sane place to keep the guidance preferences.
  QCoreApplication::setOrganizationName(QStringLiteral("Synapse"));
  QCoreApplication::setApplicationName(QStringLiteral("Synapse-AI"));

  // The application icon, set at runtime so it applies on every platform (Linux
  // X11/Wayland, Windows, macOS) without any per-platform packaging. The icon
  // carries several resolutions; the window manager picks the size it needs.
  QIcon appIcon;
  for (int sz : {16, 32, 48, 64, 128, 256, 512})
    appIcon.addFile(QStringLiteral(":/png/app-icon/icon-%1.png").arg(sz));
  app.setWindowIcon(appIcon);

  EngineBridge bridge;

  // Optional deep-link: SYNAPSE_BLUEPRINT=ocr_5x5 launches straight into a blueprint.
  const QByteArray startBlueprint = qgetenv("SYNAPSE_BLUEPRINT");
  if (!startBlueprint.isEmpty())
    bridge.loadBlueprintByName(QString::fromUtf8(startBlueprint));

  // Optional: SYNAPSE_HOLDOUT=0.3 holds out 30% of the data for validation at launch,
  // so the overfitting demo can be reproduced without clicking.
  const QByteArray holdout = qgetenv("SYNAPSE_HOLDOUT");
  if (!holdout.isEmpty()) bridge.setValFraction(holdout.toDouble());

  // Optional: SYNAPSE_OPTIMIZER=adam picks the update rule at launch.
  const QByteArray optName = qgetenv("SYNAPSE_OPTIMIZER");
  if (!optName.isEmpty()) bridge.setOptimizer(QString::fromUtf8(optName));

  // Optional: SYNAPSE_AUTOTRAIN=1 starts the supervised trainer immediately (handy for demos).
  if (!qgetenv("SYNAPSE_AUTOTRAIN").isEmpty())
    bridge.autoTrainStart();

  // Optional: SYNAPSE_LEARNSTEP=N advances the gradient-flow animation N sub-steps
  // on example 0 (handy for demos/screenshots of the backward pass).
  const QByteArray ls = qgetenv("SYNAPSE_LEARNSTEP");
  if (!ls.isEmpty())
    for (int i = 0, n = ls.toInt(); i < n; ++i) bridge.stepLearnExample(0);

  // Optional: SYNAPSE_SAVEWEIGHTS=1 writes the trained weights sidecar shortly after
  // launch (used with SYNAPSE_AUTOTRAIN to capture a trained model without clicking).
  if (!qgetenv("SYNAPSE_SAVEWEIGHTS").isEmpty())
    QTimer::singleShot(1000, &bridge, [&bridge] { bridge.saveWeights(); });

  QQmlApplicationEngine engine;
  engine.rootContext()->setContextProperty("bridge", &bridge);
  const QByteArray envTitle = qgetenv("SYNAPSE_TITLE");
  engine.rootContext()->setContextProperty(
      "appTitle", envTitle.isEmpty() ? QStringLiteral("Synapse-AI — Live Network")
                                     : QString::fromUtf8(envTitle));
  engine.rootContext()->setContextProperty("autoOpenDataManager",
                                           !qgetenv("SYNAPSE_DATAMANAGER").isEmpty());
  engine.rootContext()->setContextProperty("autoOpenCodeLab",
                                           !qgetenv("SYNAPSE_CODELAB").isEmpty());
  // Optional: SYNAPSE_SCROLL=<px> scrolls the workflow panel before a screenshot, so the
  // lower cards can be captured without resizing the window.
  engine.rootContext()->setContextProperty("startScroll", qgetenv("SYNAPSE_SCROLL").toInt());
  engine.rootContext()->setContextProperty("grabDelay",
                                           qgetenv("SYNAPSE_GRABDELAY").isEmpty()
                                               ? 1200 : qgetenv("SYNAPSE_GRABDELAY").toInt());
  engine.rootContext()->setContextProperty("grabPath",
                                           QString::fromUtf8(qgetenv("SYNAPSE_GRAB")));
  // Optional: SYNAPSE_TAB=1 opens a specific actions tab (0 Run, 1 Train, 2 Model, 3 Info).
  engine.rootContext()->setContextProperty("startTab", qgetenv("SYNAPSE_TAB").toInt());
  engine.rootContext()->setContextProperty("startAdvanced",
                                           !qgetenv("SYNAPSE_ADVANCED").isEmpty());
  engine.rootContext()->setContextProperty("selectNeuron",
                                           QString::fromUtf8(qgetenv("SYNAPSE_SELECT")));

  engine.loadFromModule("SynapseDashboard", "Main");
  if (engine.rootObjects().isEmpty())
    return -1;

  return app.exec();
}
