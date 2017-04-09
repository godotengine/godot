
Name:       harbour-godotengine

# >> macros
# << macros

#%define debug_package %{nil}

Summary:    Godot Game Engine
Version:    3.0.0
Release:    1
Group:      Qt/Qt
License:    MIT
URL:        https://godotengine.org/
Source0:    %{name}-%{version}.tar.bz2
Requires:   sailfishsilica-qt5 >= 0.10.9
BuildRequires:  pkgconfig(sailfishapp) >= 1.0.2
BuildRequires:  pkgconfig(Qt5Core)
BuildRequires:  pkgconfig(Qt5Qml)
BuildRequires:  pkgconfig(Qt5Quick)
BuildRequires:  desktop-file-utils

%description
An advanced, feature-packed, multi-platform 2D and 3D open source game engine.


%prep
%setup -q -n %{name}-%{version}

# >> setup
# << setup

%build
# >> build pre
# << build pre

# >> build post
# << build post

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_datadir}/%{name}
mkdir -p %{buildroot}%{_datadir}/%{name}/qml
install qml/*.qml %{buildroot}%{_datadir}/%{name}/qml
mkdir -p %{buildroot}%{_datadir}/applications/
install harbour-godotengine.desktop %{buildroot}%{_datadir}/applications/%{name}.desktop
install godot.sailfish.tools.32 %{buildroot}%{_bindir}/harbour-godotengine
for iconsize in 86x86 108x108 128x128 256x256
do
	mkdir -p %{buildroot}%{_datadir}/icons/hicolor/${iconsize}/apps
	install icons/${iconsize}/harbour-godotengine.png %{buildroot}%{_datadir}/icons/hicolor/${iconsize}/apps/harbour-godotengine.png
done
# >> install pre
# << install pre


# >> install post
# << install post


desktop-file-install --delete-original       \
  --dir %{buildroot}%{_datadir}/applications             \
   %{buildroot}%{_datadir}/applications/*.desktop

%files
%defattr(-,root,root,-)
%{_bindir}
%{_bindir}/harbour-godotengine
%{_datadir}/%{name}
%{_datadir}/applications/%{name}.desktop
%{_datadir}/icons/hicolor/*/apps/%{name}.png
# >> files
# << files
