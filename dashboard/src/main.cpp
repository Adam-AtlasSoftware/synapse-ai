#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "EngineBridge.h"

// The dashboard entry point. It creates the engine bridge (which builds a default
// model on the GPU), exposes it to QML as the `bridge` context property, and hands
// off to Main.qml. Everything the UI shows flows through that one object.
int main(int argc, char* argv[]) {
  QGuiApplication app(argc, argv);

  EngineBridge bridge;

  // Optional deep-link: SYNAPSE_BLUEPRINT=ocr_5x5 launches straight into a blueprint.
  const QByteArray startBlueprint = qgetenv("SYNAPSE_BLUEPRINT");
  if (!startBlueprint.isEmpty())
    bridge.loadBlueprintByName(QString::fromUtf8(startBlueprint));

  // Optional: SYNAPSE_AUTOTRAIN=1 starts training immediately (handy for demos).
  if (!qgetenv("SYNAPSE_AUTOTRAIN").isEmpty())
    bridge.trainStart();

  // Optional: SYNAPSE_LEARNSTEP=N advances the gradient-flow animation N sub-steps
  // on example 0 (handy for demos/screenshots of the backward pass).
  const QByteArray ls = qgetenv("SYNAPSE_LEARNSTEP");
  if (!ls.isEmpty())
    for (int i = 0, n = ls.toInt(); i < n; ++i) bridge.stepLearnExample(0);

  QQmlApplicationEngine engine;
  engine.rootContext()->setContextProperty("bridge", &bridge);
  const QByteArray envTitle = qgetenv("SYNAPSE_TITLE");
  engine.rootContext()->setContextProperty(
      "appTitle", envTitle.isEmpty() ? QStringLiteral("Synapse-AI — Live Network")
                                     : QString::fromUtf8(envTitle));
  engine.rootContext()->setContextProperty("autoOpenDataManager",
                                           !qgetenv("SYNAPSE_DATAMANAGER").isEmpty());
  engine.rootContext()->setContextProperty("grabPath",
                                           QString::fromUtf8(qgetenv("SYNAPSE_GRAB")));

  engine.loadFromModule("SynapseDashboard", "Main");
  if (engine.rootObjects().isEmpty())
    return -1;

  return app.exec();
}
