import QtQuick 2.2
import Sailfish.Silica 1.0

ApplicationWindow
{
    initialPage: Component { HomePage { } }
    cover: Qt.resolvedUrl("CoverPage.qml")
    allowedOrientations: Orientation.All
    _defaultPageOrientations: Orientation.All
}
