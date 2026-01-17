// Timeline Application - Main JavaScript

let currentTimeline = null;
let currentEvents = [];
let selectedEventIndex = 0;
let isAutoPlaying = false;
let autoPlayInterval = null;
let zoomLevel = 1;
let isDragging = false;
let scrubberPosition = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeScrollIndicator();
    initializeAnimations();
});

// Scroll Indicator
function initializeScrollIndicator() {
    const scrollIndicator = document.querySelector('.scroll-indicator');
    if (scrollIndicator) {
        scrollIndicator.addEventListener('click', () => {
            document.querySelector('.timeline-selector').scrollIntoView({ behavior: 'smooth' });
        });
    }
}

// Initialize Animations
function initializeAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.timeline-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'all 0.6s ease';
        observer.observe(card);
    });
}

// Load Timeline
async function loadTimeline(timelineId) {
    try {
        const response = await fetch(`/api/timeline/${timelineId}`);
        const data = await response.json();
        
        currentTimeline = data;
        currentEvents = data.events;
        selectedEventIndex = 0;
        zoomLevel = 1;
        
        // Hide selector and show viewer
        document.querySelector('.timeline-selector').style.display = 'none';
        const viewer = document.getElementById('timelineViewer');
        viewer.style.display = 'block';
        
        // Update header
        document.getElementById('timelineTitle').textContent = data.title;
        document.getElementById('timelineDescription').textContent = data.description;
        
        // Render timeline
        renderTimeline();
        
        // Initialize controls
        initializeControls();
        
        // Show first event
        showEventDetails(0);
        
        // Scroll to timeline
        viewer.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Error loading timeline:', error);
    }
}

// Render Timeline
function renderTimeline() {
    const eventsContainer = document.getElementById('timelineEvents');
    eventsContainer.innerHTML = '';
    
    currentEvents.forEach((event, index) => {
        const eventElement = document.createElement('div');
        eventElement.className = 'timeline-event';
        if (index === selectedEventIndex) {
            eventElement.classList.add('active');
        }
        
        eventElement.innerHTML = `
            <div class="event-dot"></div>
            <div class="event-year-label">${event.year}</div>
            <div class="event-title-label">${event.title}</div>
        `;
        
        eventElement.addEventListener('click', () => showEventDetails(index));
        eventsContainer.appendChild(eventElement);
    });
    
    // Update progress bar
    updateProgress();
    
    // Initialize scrubber
    initializeScrubber();
}

// Update Progress Bar
function updateProgress() {
    const progress = document.getElementById('timelineProgress');
    const percentage = (selectedEventIndex / (currentEvents.length - 1)) * 100;
    progress.style.width = `${percentage}%`;
}

// Initialize Scrubber
function initializeScrubber() {
    const scrubber = document.getElementById('timelineScrubber');
    const handle = scrubber.querySelector('.scrubber-handle');
    const container = document.getElementById('timelineContainer');
    const track = document.getElementById('timelineTrack');
    
    // Position scrubber
    updateScrubberPosition();
    
    // Drag functionality
    handle.addEventListener('mousedown', startDragging);
    handle.addEventListener('touchstart', startDragging);
    
    document.addEventListener('mousemove', drag);
    document.addEventListener('touchmove', drag);
    
    document.addEventListener('mouseup', stopDragging);
    document.addEventListener('touchend', stopDragging);
}

function startDragging(e) {
    e.preventDefault();
    isDragging = true;
    document.querySelector('.scrubber-handle').style.cursor = 'grabbing';
}

function drag(e) {
    if (!isDragging) return;
    
    const container = document.getElementById('timelineContainer');
    const track = document.getElementById('timelineTrack');
    const rect = track.getBoundingClientRect();
    
    let clientX;
    if (e.type === 'touchmove') {
        clientX = e.touches[0].clientX;
    } else {
        clientX = e.clientX;
    }
    
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    
    // Calculate nearest event
    const eventIndex = Math.round(percentage * (currentEvents.length - 1));
    
    if (eventIndex !== selectedEventIndex) {
        selectedEventIndex = eventIndex;
        updateActiveEvent();
        showEventDetails(eventIndex);
    }
}

function stopDragging() {
    isDragging = false;
    document.querySelector('.scrubber-handle').style.cursor = 'grab';
}

function updateScrubberPosition() {
    const scrubber = document.getElementById('timelineScrubber');
    const track = document.getElementById('timelineTrack');
    const percentage = selectedEventIndex / (currentEvents.length - 1);
    const position = percentage * track.offsetWidth;
    scrubber.style.left = `${position}px`;
}

function updateActiveEvent() {
    document.querySelectorAll('.timeline-event').forEach((el, idx) => {
        el.classList.toggle('active', idx === selectedEventIndex);
    });
    updateProgress();
    updateScrubberPosition();
}

// Show Event Details
function showEventDetails(index) {
    selectedEventIndex = index;
    const event = currentEvents[index];
    
    // Update active state
    updateActiveEvent();
    
    // Update details panel
    const detailsPanel = document.getElementById('eventDetails');
    document.getElementById('eventImage').style.backgroundImage = `url('${event.image}')`;
    document.getElementById('eventYear').textContent = event.year;
    document.getElementById('eventTitle').textContent = event.title;
    document.getElementById('eventDescription').textContent = event.description;
    
    // Update tags
    const tagsContainer = document.getElementById('eventTags');
    tagsContainer.innerHTML = event.tags.map(tag => 
        `<span class="event-tag">${tag}</span>`
    ).join('');
    
    // Show panel with animation
    requestAnimationFrame(() => {
        detailsPanel.classList.add('active');
    });
    
    // Scroll event into view
    const eventElements = document.querySelectorAll('.timeline-event');
    if (eventElements[index]) {
        eventElements[index].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
}

// Close Event Details
function closeEventDetails() {
    document.getElementById('eventDetails').classList.remove('active');
}

// Close Timeline
function closeTimeline() {
    // Stop auto-play if active
    if (isAutoPlaying) {
        toggleAutoPlay();
    }
    
    // Hide viewer and show selector
    document.getElementById('timelineViewer').style.display = 'none';
    const selector = document.querySelector('.timeline-selector');
    selector.style.display = 'block';
    
    // Close details panel
    closeEventDetails();
    
    // Scroll to selector
    selector.scrollIntoView({ behavior: 'smooth' });
}

// Initialize Controls
function initializeControls() {
    // Zoom controls
    document.getElementById('zoomIn').addEventListener('click', () => {
        zoomLevel = Math.min(2, zoomLevel + 0.2);
        applyZoom();
    });
    
    document.getElementById('zoomOut').addEventListener('click', () => {
        zoomLevel = Math.max(0.5, zoomLevel - 0.2);
        applyZoom();
    });
    
    // Auto-play control
    document.getElementById('autoPlay').addEventListener('click', toggleAutoPlay);
    
    // Fullscreen control
    document.getElementById('fullscreen').addEventListener('click', toggleFullscreen);
    
    // Keyboard navigation
    document.addEventListener('keydown', handleKeyboard);
}

// Apply Zoom
function applyZoom() {
    const track = document.getElementById('timelineTrack');
    track.style.transform = `scale(${zoomLevel})`;
    track.style.transformOrigin = 'center';
}

// Toggle Auto-Play
function toggleAutoPlay() {
    const btn = document.getElementById('autoPlay');
    
    if (isAutoPlaying) {
        // Stop auto-play
        isAutoPlaying = false;
        clearInterval(autoPlayInterval);
        btn.classList.remove('active');
        btn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
        `;
    } else {
        // Start auto-play
        isAutoPlaying = true;
        btn.classList.add('active');
        btn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
            </svg>
        `;
        
        autoPlayInterval = setInterval(() => {
            selectedEventIndex = (selectedEventIndex + 1) % currentEvents.length;
            showEventDetails(selectedEventIndex);
        }, 3000);
    }
}

// Toggle Fullscreen
function toggleFullscreen() {
    const viewer = document.getElementById('timelineViewer');
    const btn = document.getElementById('fullscreen');
    
    if (!document.fullscreenElement) {
        viewer.requestFullscreen().catch(err => {
            console.error('Error attempting to enable fullscreen:', err);
        });
        btn.classList.add('active');
    } else {
        document.exitFullscreen();
        btn.classList.remove('active');
    }
}

// Keyboard Navigation
function handleKeyboard(e) {
    if (!currentTimeline) return;
    
    switch(e.key) {
        case 'ArrowLeft':
            e.preventDefault();
            if (selectedEventIndex > 0) {
                showEventDetails(selectedEventIndex - 1);
            }
            break;
        case 'ArrowRight':
            e.preventDefault();
            if (selectedEventIndex < currentEvents.length - 1) {
                showEventDetails(selectedEventIndex + 1);
            }
            break;
        case 'Escape':
            e.preventDefault();
            if (document.getElementById('eventDetails').classList.contains('active')) {
                closeEventDetails();
            } else {
                closeTimeline();
            }
            break;
        case ' ':
            e.preventDefault();
            toggleAutoPlay();
            break;
        case 'f':
            e.preventDefault();
            toggleFullscreen();
            break;
        case '+':
        case '=':
            e.preventDefault();
            zoomLevel = Math.min(2, zoomLevel + 0.2);
            applyZoom();
            break;
        case '-':
        case '_':
            e.preventDefault();
            zoomLevel = Math.max(0.5, zoomLevel - 0.2);
            applyZoom();
            break;
    }
}

// Mouse Wheel Zoom
document.addEventListener('wheel', (e) => {
    if (!currentTimeline) return;
    
    const timelineContainer = document.getElementById('timelineContainer');
    if (timelineContainer && timelineContainer.contains(e.target)) {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            
            if (e.deltaY < 0) {
                // Zoom in
                zoomLevel = Math.min(2, zoomLevel + 0.05);
            } else {
                // Zoom out
                zoomLevel = Math.max(0.5, zoomLevel - 0.05);
            }
            
            applyZoom();
        }
    }
}, { passive: false });

// Smooth scroll for timeline container
document.addEventListener('DOMContentLoaded', () => {
    const timelineContainer = document.getElementById('timelineContainer');
    if (timelineContainer) {
        let isScrolling = false;
        let scrollTimeout;
        
        timelineContainer.addEventListener('scroll', () => {
            isScrolling = true;
            clearTimeout(scrollTimeout);
            
            scrollTimeout = setTimeout(() => {
                isScrolling = false;
            }, 150);
        });
    }
});

// Touch gestures for mobile
let touchStartX = 0;
let touchEndX = 0;

document.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
}, { passive: true });

document.addEventListener('touchend', (e) => {
    if (!currentTimeline) return;
    
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
}, { passive: true });

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;
    
    if (Math.abs(diff) > swipeThreshold) {
        if (diff > 0 && selectedEventIndex < currentEvents.length - 1) {
            // Swipe left - next event
            showEventDetails(selectedEventIndex + 1);
        } else if (diff < 0 && selectedEventIndex > 0) {
            // Swipe right - previous event
            showEventDetails(selectedEventIndex - 1);
        }
    }
}

// Prevent default behavior for some keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '-' || e.key === '=')) {
        if (currentTimeline) {
            e.preventDefault();
        }
    }
});

// Window resize handler
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        if (currentTimeline) {
            updateScrubberPosition();
        }
    }, 250);
});

console.log('Timeline App initialized! 🚀');
console.log('Keyboard shortcuts:');
console.log('  ← → : Navigate between events');
console.log('  Space : Toggle auto-play');
console.log('  F : Toggle fullscreen');
console.log('  +/- : Zoom in/out');
console.log('  Esc : Close timeline/event details');
console.log('  Ctrl+Scroll : Zoom with mouse wheel');
