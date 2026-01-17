# 🎨 Wallie - iOS Wallpaper Customizer

A beautiful web-based iOS wallpaper customizer that mimics the iOS lock screen experience with interactive hold and swipe gestures.

## ✨ Features

- **🔒 iOS Lock Screen Mockup** - Realistic lock screen design with status bar, time, date, and widgets
- **👆 Hold & Swipe Gestures** - Hold for 800ms, then swipe up/left to change wallpapers
- **📱 Responsive Design** - Optimized for Mobile, iPad, and Web views
- **🎨 16 Beautiful Wallpapers** - Curated collection of gradient wallpapers
- **⚡ Smooth Animations** - Fluid transitions and visual feedback
- **🎯 Touch-friendly** - Works on both touch devices and desktop

## 🚀 Quick Start

Simply open `index.html` in your web browser:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve

# Or just open the file
open index.html
```

Then navigate to `http://localhost:8000` in your browser.

## 🎮 How to Use

### Touch/Mouse Interaction

1. **Hold** on the lock screen for 800ms (a circular progress indicator will appear)
2. **Swipe** in any direction:
   - **Swipe Up** ⬆️ - Next wallpaper (slide up animation)
   - **Swipe Down** ⬇️ - Previous wallpaper (slide up animation)
   - **Swipe Left** ⬅️ - Next wallpaper (slide left animation)
   - **Swipe Right** ➡️ - Previous wallpaper (slide left animation)

### Gallery Interaction

- Click/tap any wallpaper in the gallery to instantly apply it

### Device Views

Toggle between different device layouts:
- 📱 **Mobile** - iPhone-style portrait view
- 📱 **iPad** - Tablet landscape view
- 💻 **Web** - Full-screen desktop view

### Keyboard Shortcuts (Desktop Only)

- `↑` / `↓` - Navigate wallpapers (vertical animation)
- `←` / `→` - Navigate wallpapers (horizontal animation)
- `1` - Switch to Mobile view
- `2` - Switch to iPad view
- `3` - Switch to Web view

## 🏗️ Project Structure

```
Wallie App/
├── index.html       # Main HTML structure
├── styles.css       # iOS-style CSS with responsive design
├── script.js        # Gesture handling and app logic
└── README.md        # Documentation
```

## 🎨 Wallpaper Collection

The app includes 16 beautiful gradient wallpapers:
- Ocean Gradient
- Sunset
- Forest
- Aurora
- Lavender
- Deep Ocean
- Berry
- Fire
- Sky
- Purple Dream
- Mint
- Rose
- Cosmic
- Peach
- Emerald
- Neon

## 🛠️ Technical Details

### Technologies Used

- **HTML5** - Semantic markup
- **CSS3** - Modern styling with gradients, animations, and backdrop-filter
- **Vanilla JavaScript** - No dependencies, pure ES6+

### Key Features Implemented

1. **Touch Event Handling**
   - Supports both touch and mouse events
   - Precise gesture detection (hold duration + swipe direction)
   - Visual feedback with progress indicator

2. **Responsive Design**
   - CSS Grid and Flexbox
   - Media queries for different breakpoints
   - Aspect ratio preservation across devices

3. **Animations**
   - CSS transitions and keyframe animations
   - Dynamic class toggling for wallpaper changes
   - Smooth state transitions

4. **iOS Styling**
   - San Francisco font stack (-apple-system)
   - Backdrop blur effects
   - Status bar simulation
   - Lock screen widgets

## 📱 Browser Compatibility

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## 🔮 Future Enhancements

- [ ] Add custom wallpaper upload
- [ ] Save favorite wallpapers
- [ ] Share wallpaper selections
- [ ] Add more wallpaper categories
- [ ] Implement parallax effects
- [ ] Add haptic feedback for supported devices
- [ ] Dark/Light mode toggle
- [ ] Export as mobile app (PWA)

## 📄 License

MIT License - Feel free to use this project for personal or commercial purposes.

## 👨‍💻 Development

To contribute or modify:

1. Clone the repository
2. Make your changes
3. Test across different devices/browsers
4. Submit a pull request

## 🎯 Project Goals

This project was created to demonstrate:
- Modern web development techniques
- iOS-inspired UI/UX design
- Touch gesture implementation
- Responsive design patterns
- Smooth animations and transitions

## 🙏 Acknowledgments

- Inspired by iOS 16+ Lock Screen customization
- Gradient palettes from various design resources
- Icons and UI patterns from Apple Human Interface Guidelines

---

Built with ❤️ for the Mini Sass project
