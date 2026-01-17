// Wallpaper Collection - Beautiful gradient and image wallpapers
const wallpapers = [
    {
        id: 1,
        name: 'Ocean Gradient',
        gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    {
        id: 2,
        name: 'Sunset',
        gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    },
    {
        id: 3,
        name: 'Forest',
        gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    },
    {
        id: 4,
        name: 'Aurora',
        gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
    },
    {
        id: 5,
        name: 'Lavender',
        gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
    },
    {
        id: 6,
        name: 'Deep Ocean',
        gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)'
    },
    {
        id: 7,
        name: 'Berry',
        gradient: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'
    },
    {
        id: 8,
        name: 'Fire',
        gradient: 'linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%)'
    },
    {
        id: 9,
        name: 'Sky',
        gradient: 'linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)'
    },
    {
        id: 10,
        name: 'Purple Dream',
        gradient: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)'
    },
    {
        id: 11,
        name: 'Mint',
        gradient: 'linear-gradient(135deg, #d299c2 0%, #fef9d7 100%)'
    },
    {
        id: 12,
        name: 'Rose',
        gradient: 'linear-gradient(135deg, #f77062 0%, #fe5196 100%)'
    },
    {
        id: 13,
        name: 'Cosmic',
        gradient: 'linear-gradient(135deg, #6a11cb 0%, #2575fc 100%)'
    },
    {
        id: 14,
        name: 'Peach',
        gradient: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)'
    },
    {
        id: 15,
        name: 'Emerald',
        gradient: 'linear-gradient(135deg, #56ab2f 0%, #a8e063 100%)'
    },
    {
        id: 16,
        name: 'Neon',
        gradient: 'linear-gradient(135deg, #eecda3 0%, #ef629f 100%)'
    }
];

// App State
let currentWallpaperIndex = 0;
let currentDevice = 'mobile';
let isHolding = false;
let holdTimer = null;
let holdProgress = 0;
const HOLD_DURATION = 800; // milliseconds

// DOM Elements
const lockScreen = document.getElementById('lockScreen');
const wallpaperBg = document.getElementById('wallpaperBg');
const galleryGrid = document.getElementById('galleryGrid');
const deviceButtons = document.querySelectorAll('.device-btn');
const deviceFrame = document.querySelector('.device-frame');
const holdIndicator = document.getElementById('holdIndicator');
const gestureHint = document.getElementById('gestureHint');
const navArrowUp = document.getElementById('navArrowUp');
const navArrowLeft = document.getElementById('navArrowLeft');

// Touch/Mouse tracking
let touchStartX = 0;
let touchStartY = 0;
let touchCurrentX = 0;
let touchCurrentY = 0;

// Initialize App
function init() {
    updateTime();
    setInterval(updateTime, 1000);
    
    renderGallery();
    setWallpaper(0);
    
    setupEventListeners();
}

// Update Time Display
function updateTime() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const timeString = `${hours}:${minutes.toString().padStart(2, '0')}`;
    
    document.getElementById('currentTime').textContent = timeString;
    document.getElementById('largeTime').textContent = timeString;
    
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    
    const dayName = days[now.getDay()];
    const monthName = months[now.getMonth()];
    const date = now.getDate();
    
    document.getElementById('currentDay').textContent = `${dayName}, ${monthName} ${date}`;
}

// Render Wallpaper Gallery
function renderGallery() {
    galleryGrid.innerHTML = '';
    
    wallpapers.forEach((wallpaper, index) => {
        const item = document.createElement('div');
        item.className = 'wallpaper-item';
        if (index === currentWallpaperIndex) {
            item.classList.add('active');
        }
        
        item.style.background = wallpaper.gradient;
        item.setAttribute('data-index', index);
        item.title = wallpaper.name;
        
        item.addEventListener('click', () => {
            setWallpaper(index);
        });
        
        galleryGrid.appendChild(item);
    });
}

// Set Wallpaper
function setWallpaper(index, animation = null) {
    if (index < 0) index = wallpapers.length - 1;
    if (index >= wallpapers.length) index = 0;
    
    currentWallpaperIndex = index;
    const wallpaper = wallpapers[index];
    
    // Add animation class if specified
    if (animation) {
        wallpaperBg.classList.remove('slide-up', 'slide-left');
        // Force reflow
        void wallpaperBg.offsetWidth;
        wallpaperBg.classList.add(animation);
        
        setTimeout(() => {
            wallpaperBg.classList.remove(animation);
        }, 400);
    }
    
    wallpaperBg.style.background = wallpaper.gradient;
    
    // Update gallery active state
    document.querySelectorAll('.wallpaper-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });
}

// Setup Event Listeners
function setupEventListeners() {
    // Device selector
    deviceButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const device = e.currentTarget.getAttribute('data-device');
            switchDevice(device);
        });
    });
    
    // Touch/Mouse events for hold and swipe
    lockScreen.addEventListener('mousedown', handlePointerDown);
    lockScreen.addEventListener('touchstart', handlePointerDown, { passive: false });
    
    lockScreen.addEventListener('mousemove', handlePointerMove);
    lockScreen.addEventListener('touchmove', handlePointerMove, { passive: false });
    
    lockScreen.addEventListener('mouseup', handlePointerUp);
    lockScreen.addEventListener('touchend', handlePointerUp);
    
    lockScreen.addEventListener('mouseleave', handlePointerUp);
    lockScreen.addEventListener('touchcancel', handlePointerUp);
    
    // Prevent context menu on long press
    lockScreen.addEventListener('contextmenu', (e) => e.preventDefault());
}

// Handle Pointer Down (Mouse/Touch Start)
function handlePointerDown(e) {
    e.preventDefault();
    
    const pointer = e.touches ? e.touches[0] : e;
    touchStartX = pointer.clientX;
    touchStartY = pointer.clientY;
    touchCurrentX = touchStartX;
    touchCurrentY = touchStartY;
    
    isHolding = true;
    holdProgress = 0;
    
    // Hide gesture hint after first interaction
    gestureHint.style.display = 'none';
    
    // Show hold indicator
    holdIndicator.classList.add('active');
    
    // Start hold timer
    startHoldTimer();
}

// Handle Pointer Move (Mouse/Touch Move)
function handlePointerMove(e) {
    if (!isHolding) return;
    
    e.preventDefault();
    
    const pointer = e.touches ? e.touches[0] : e;
    touchCurrentX = pointer.clientX;
    touchCurrentY = pointer.clientY;
    
    const deltaX = touchCurrentX - touchStartX;
    const deltaY = touchCurrentY - touchStartY;
    
    // Show directional arrows based on movement
    if (Math.abs(deltaY) > 30 && Math.abs(deltaY) > Math.abs(deltaX)) {
        navArrowUp.classList.add('show');
        navArrowLeft.classList.remove('show');
    } else if (Math.abs(deltaX) > 30 && Math.abs(deltaX) > Math.abs(deltaY)) {
        navArrowLeft.classList.add('show');
        navArrowUp.classList.remove('show');
    }
}

// Handle Pointer Up (Mouse/Touch End)
function handlePointerUp(e) {
    if (!isHolding) return;
    
    const deltaX = touchCurrentX - touchStartX;
    const deltaY = touchCurrentY - touchStartY;
    
    const SWIPE_THRESHOLD = 50;
    
    // Check if hold was completed
    if (holdProgress >= 100) {
        // Determine swipe direction
        if (Math.abs(deltaY) > Math.abs(deltaX) && Math.abs(deltaY) > SWIPE_THRESHOLD) {
            // Vertical swipe
            if (deltaY < 0) {
                // Swipe up
                changeWallpaper('next', 'slide-up');
            } else {
                // Swipe down
                changeWallpaper('prev', 'slide-up');
            }
        } else if (Math.abs(deltaX) > SWIPE_THRESHOLD) {
            // Horizontal swipe
            if (deltaX < 0) {
                // Swipe left
                changeWallpaper('next', 'slide-left');
            } else {
                // Swipe right
                changeWallpaper('prev', 'slide-left');
            }
        }
    }
    
    // Reset state
    isHolding = false;
    holdProgress = 0;
    clearInterval(holdTimer);
    holdIndicator.classList.remove('active');
    navArrowUp.classList.remove('show');
    navArrowLeft.classList.remove('show');
    
    updateHoldProgress();
}

// Start Hold Timer
function startHoldTimer() {
    const updateInterval = 50;
    const progressIncrement = (100 / HOLD_DURATION) * updateInterval;
    
    holdTimer = setInterval(() => {
        if (!isHolding) {
            clearInterval(holdTimer);
            return;
        }
        
        holdProgress += progressIncrement;
        
        if (holdProgress >= 100) {
            holdProgress = 100;
            // Add haptic feedback simulation (visual)
            holdIndicator.style.transform = 'translate(-50%, -50%) scale(1.1)';
            setTimeout(() => {
                holdIndicator.style.transform = 'translate(-50%, -50%) scale(1)';
            }, 100);
        }
        
        updateHoldProgress();
    }, updateInterval);
}

// Update Hold Progress Visual
function updateHoldProgress() {
    const progressRing = holdIndicator.querySelector('.progress-ring');
    const circumference = 2 * Math.PI * 45;
    const offset = circumference - (holdProgress / 100) * circumference;
    progressRing.style.strokeDashoffset = offset;
}

// Change Wallpaper
function changeWallpaper(direction, animation) {
    let newIndex = currentWallpaperIndex;
    
    if (direction === 'next') {
        newIndex = (currentWallpaperIndex + 1) % wallpapers.length;
    } else if (direction === 'prev') {
        newIndex = currentWallpaperIndex - 1;
        if (newIndex < 0) newIndex = wallpapers.length - 1;
    }
    
    setWallpaper(newIndex, animation);
}

// Switch Device View
function switchDevice(device) {
    currentDevice = device;
    
    // Update button states
    deviceButtons.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-device') === device);
    });
    
    // Update device frame class
    deviceFrame.classList.remove('mobile-frame', 'tablet-frame', 'web-frame');
    deviceFrame.classList.add(`${device}-frame`);
}

// Keyboard shortcuts for testing
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp') {
        changeWallpaper('next', 'slide-up');
    } else if (e.key === 'ArrowDown') {
        changeWallpaper('prev', 'slide-up');
    } else if (e.key === 'ArrowLeft') {
        changeWallpaper('prev', 'slide-left');
    } else if (e.key === 'ArrowRight') {
        changeWallpaper('next', 'slide-left');
    } else if (e.key === '1') {
        switchDevice('mobile');
    } else if (e.key === '2') {
        switchDevice('tablet');
    } else if (e.key === '3') {
        switchDevice('web');
    }
});

// Handle window resize
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        // Re-render if needed
        renderGallery();
    }, 250);
});

// Prevent pull-to-refresh on mobile
document.body.addEventListener('touchmove', (e) => {
    if (e.target.closest('.lock-screen')) {
        e.preventDefault();
    }
}, { passive: false });

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Add smooth scroll behavior
document.documentElement.style.scrollBehavior = 'smooth';

// Log initialization
console.log('🎨 Wallie App initialized');
console.log('💡 Tips:');
console.log('  - Hold on the lock screen for 800ms, then swipe');
console.log('  - Use arrow keys for quick testing');
console.log('  - Press 1, 2, or 3 to switch device views');
console.log(`  - ${wallpapers.length} wallpapers loaded`);
