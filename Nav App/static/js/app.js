// State management
let destinations = [];
let routes = [];
let lists = [];
let currentRoute = [];
let isCreatingRoute = false;
let draggedElement = null;
let draggedListItem = null;

// API Base URL
const API_BASE = '/api';

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    loadDestinations();
    loadLists();
});

function initializeApp() {
    // Set up view toggle
    const viewButtons = document.querySelectorAll('.toggle-btn');
    viewButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const viewType = btn.dataset.view;
            switchView(viewType);
            
            viewButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

function switchView(viewType) {
    const views = document.querySelectorAll('.view-container');
    views.forEach(view => view.classList.remove('active'));
    
    const activeView = document.getElementById(`${viewType}View`);
    if (activeView) {
        activeView.classList.add('active');
        
        if (viewType === 'list') {
            renderPlannerList();
        } else if (viewType === 'shared') {
            renderSharedLists();
        }
    }
}

function setupEventListeners() {
    // Add destination button
    document.getElementById('addDestinationBtn').addEventListener('click', () => {
        document.getElementById('addDestinationModal').classList.add('active');
    });
    
    // Close destination modal
    document.getElementById('closeDestinationModal').addEventListener('click', closeDestinationModal);
    document.getElementById('cancelDestBtn').addEventListener('click', closeDestinationModal);
    
    // Save destination
    document.getElementById('saveDestBtn').addEventListener('click', saveDestination);
    
    // Emoji picker
    document.querySelectorAll('.emoji-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            document.getElementById('destEmoji').value = btn.dataset.emoji;
        });
    });
    
    // Color picker
    document.querySelectorAll('.color-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            document.getElementById('destColor').value = btn.dataset.color;
        });
    });
    
    // Create route button
    document.getElementById('createRouteBtn').addEventListener('click', toggleRouteCreation);
    
    // Clear routes button
    document.getElementById('clearRoutesBtn').addEventListener('click', clearRoutes);
    
    // Create list button
    document.getElementById('createListBtn').addEventListener('click', () => {
        document.getElementById('createListModal').classList.add('active');
    });
    
    // Close list modal
    document.getElementById('closeListModal').addEventListener('click', closeListModal);
    document.getElementById('cancelListBtn').addEventListener('click', closeListModal);
    
    // Save list
    document.getElementById('saveListBtn').addEventListener('click', saveList);
}

// Destination Management
async function loadDestinations() {
    try {
        const response = await fetch(`${API_BASE}/destinations`);
        destinations = await response.json();
        renderDestinations();
    } catch (error) {
        console.error('Error loading destinations:', error);
    }
}

async function saveDestination() {
    const name = document.getElementById('destName').value.trim();
    const emoji = document.getElementById('destEmoji').value;
    const color = document.getElementById('destColor').value;
    const category = document.getElementById('destCategory').value;
    
    if (!name) {
        alert('Please enter a destination name');
        return;
    }
    
    const newDestination = {
        name,
        emoji,
        color,
        category,
        x: Math.random() * 60 + 20, // Random position between 20-80%
        y: Math.random() * 60 + 20
    };
    
    try {
        const response = await fetch(`${API_BASE}/destinations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newDestination)
        });
        
        const destination = await response.json();
        destinations.push(destination);
        renderDestinations();
        closeDestinationModal();
        resetDestinationForm();
    } catch (error) {
        console.error('Error saving destination:', error);
    }
}

function renderDestinations() {
    const canvas = document.getElementById('mapCanvas');
    
    // Remove existing destinations (keep current location and SVG)
    const existingDests = canvas.querySelectorAll('.destination-item');
    existingDests.forEach(dest => dest.remove());
    
    destinations.forEach(dest => {
        const destElement = createDestinationElement(dest);
        canvas.appendChild(destElement);
    });
}

function createDestinationElement(dest) {
    const wrapper = document.createElement('div');
    wrapper.className = 'destination-item';
    wrapper.style.left = `${dest.x}%`;
    wrapper.style.top = `${dest.y}%`;
    wrapper.dataset.id = dest.id;
    
    const bubble = document.createElement('div');
    bubble.className = 'destination-bubble';
    bubble.style.background = dest.color;
    
    const emoji = document.createElement('span');
    emoji.className = 'bubble-emoji';
    emoji.textContent = dest.emoji;
    
    const label = document.createElement('div');
    label.className = 'bubble-label';
    label.textContent = dest.name;
    
    bubble.appendChild(emoji);
    wrapper.appendChild(bubble);
    wrapper.appendChild(label);
    
    // Make draggable
    makeDraggable(wrapper, dest);
    
    // Click handler for route creation
    bubble.addEventListener('click', () => {
        if (isCreatingRoute) {
            addToRoute(dest);
        }
    });
    
    return wrapper;
}

function makeDraggable(element, dest) {
    let isDragging = false;
    let startX, startY;
    
    const bubble = element.querySelector('.destination-bubble');
    
    bubble.addEventListener('mousedown', startDrag);
    
    function startDrag(e) {
        isDragging = true;
        bubble.classList.add('dragging');
        
        const rect = element.getBoundingClientRect();
        const canvas = document.getElementById('mapCanvas');
        const canvasRect = canvas.getBoundingClientRect();
        
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        
        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', stopDrag);
    }
    
    function drag(e) {
        if (!isDragging) return;
        
        const canvas = document.getElementById('mapCanvas');
        const canvasRect = canvas.getBoundingClientRect();
        
        const x = ((e.clientX - canvasRect.left - startX) / canvasRect.width) * 100;
        const y = ((e.clientY - canvasRect.top - startY) / canvasRect.height) * 100;
        
        // Constrain to canvas
        const constrainedX = Math.max(5, Math.min(95, x));
        const constrainedY = Math.max(5, Math.min(95, y));
        
        element.style.left = `${constrainedX}%`;
        element.style.top = `${constrainedY}%`;
        
        dest.x = constrainedX;
        dest.y = constrainedY;
        
        updateRoutes();
    }
    
    async function stopDrag() {
        if (!isDragging) return;
        isDragging = false;
        bubble.classList.remove('dragging');
        
        document.removeEventListener('mousemove', drag);
        document.removeEventListener('mouseup', stopDrag);
        
        // Save position to server
        try {
            await fetch(`${API_BASE}/destinations/${dest.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: dest.x, y: dest.y })
            });
        } catch (error) {
            console.error('Error updating destination:', error);
        }
    }
}

function closeDestinationModal() {
    document.getElementById('addDestinationModal').classList.remove('active');
}

function resetDestinationForm() {
    document.getElementById('destName').value = '';
    document.getElementById('destEmoji').value = '📍';
    document.getElementById('destColor').value = '#4ECDC4';
    document.getElementById('destCategory').value = 'general';
    
    document.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('selected'));
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('selected'));
}

// Route Management
function toggleRouteCreation() {
    isCreatingRoute = !isCreatingRoute;
    const btn = document.getElementById('createRouteBtn');
    
    if (isCreatingRoute) {
        btn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        btn.style.color = 'white';
        currentRoute = [];
    } else {
        btn.style.background = 'white';
        btn.style.color = '#667eea';
        
        if (currentRoute.length > 1) {
            saveRoute();
        }
        currentRoute = [];
    }
}

function addToRoute(dest) {
    currentRoute.push(dest);
    updateRoutes();
}

function updateRoutes() {
    const svg = document.getElementById('routeSvg');
    svg.innerHTML = '';
    
    if (currentRoute.length > 1) {
        drawRoute(currentRoute, '#2196F3', true);
    }
    
    routes.forEach(route => {
        const routeDests = route.destinations.map(id => 
            destinations.find(d => d.id === id)
        ).filter(d => d);
        
        if (routeDests.length > 1) {
            drawRoute(routeDests, route.color, false);
        }
    });
}

function drawRoute(destinations, color, animated = false) {
    const svg = document.getElementById('routeSvg');
    const canvas = document.getElementById('mapCanvas');
    const canvasRect = canvas.getBoundingClientRect();
    
    for (let i = 0; i < destinations.length - 1; i++) {
        const start = destinations[i];
        const end = destinations[i + 1];
        
        const x1 = (start.x / 100) * canvasRect.width + 40; // +40 for bubble center
        const y1 = (start.y / 100) * canvasRect.height + 40;
        const x2 = (end.x / 100) * canvasRect.width + 40;
        const y2 = (end.y / 100) * canvasRect.height + 40;
        
        // Create curved path
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const offset = 30;
        const cpX = midX - dy * offset / Math.sqrt(dx * dx + dy * dy);
        const cpY = midY + dx * offset / Math.sqrt(dx * dx + dy * dy);
        
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M ${x1} ${y1} Q ${cpX} ${cpY} ${x2} ${y2}`);
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', animated ? '4' : '3');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke-linecap', 'round');
        
        if (animated) {
            path.setAttribute('stroke-dasharray', '10,5');
            path.style.animation = 'dash 1s linear infinite';
        }
        
        svg.appendChild(path);
        
        // Add arrow
        const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        const angle = Math.atan2(y2 - cpY, x2 - cpX);
        const arrowSize = 10;
        
        const points = [
            [x2, y2],
            [x2 - arrowSize * Math.cos(angle - Math.PI / 6), y2 - arrowSize * Math.sin(angle - Math.PI / 6)],
            [x2 - arrowSize * Math.cos(angle + Math.PI / 6), y2 - arrowSize * Math.sin(angle + Math.PI / 6)]
        ];
        
        arrow.setAttribute('points', points.map(p => p.join(',')).join(' '));
        arrow.setAttribute('fill', color);
        svg.appendChild(arrow);
    }
}

async function saveRoute() {
    if (currentRoute.length < 2) return;
    
    const newRoute = {
        name: `Route ${routes.length + 1}`,
        destinations: currentRoute.map(d => d.id),
        color: '#2196F3'
    };
    
    try {
        const response = await fetch(`${API_BASE}/routes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newRoute)
        });
        
        const route = await response.json();
        routes.push(route);
        updateRoutes();
    } catch (error) {
        console.error('Error saving route:', error);
    }
}

function clearRoutes() {
    routes = [];
    currentRoute = [];
    isCreatingRoute = false;
    
    const btn = document.getElementById('createRouteBtn');
    btn.style.background = 'white';
    btn.style.color = '#667eea';
    
    updateRoutes();
}

// List Management
async function loadLists() {
    try {
        const response = await fetch(`${API_BASE}/lists`);
        lists = await response.json();
    } catch (error) {
        console.error('Error loading lists:', error);
    }
}

async function saveList() {
    const name = document.getElementById('listName').value.trim();
    const description = document.getElementById('listDescription').value.trim();
    const category = document.getElementById('listCategory').value;
    const shared = document.getElementById('listShared').checked;
    
    if (!name) {
        alert('Please enter a list name');
        return;
    }
    
    const newList = {
        name,
        description,
        category,
        shared,
        destinations: [],
        creator: 'You'
    };
    
    try {
        const response = await fetch(`${API_BASE}/lists`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newList)
        });
        
        const list = await response.json();
        lists.push(list);
        closeListModal();
        resetListForm();
        renderSharedLists();
    } catch (error) {
        console.error('Error saving list:', error);
    }
}

function closeListModal() {
    document.getElementById('createListModal').classList.remove('active');
}

function resetListForm() {
    document.getElementById('listName').value = '';
    document.getElementById('listDescription').value = '';
    document.getElementById('listCategory').value = 'food';
    document.getElementById('listShared').checked = false;
}

function renderSharedLists() {
    const grid = document.getElementById('listsGrid');
    grid.innerHTML = '';
    
    const sharedLists = lists.filter(list => list.shared);
    
    if (sharedLists.length === 0) {
        grid.innerHTML = '<p style="text-align: center; color: #666;">No shared lists yet. Create one to get started!</p>';
        return;
    }
    
    sharedLists.forEach(list => {
        const card = document.createElement('div');
        card.className = 'list-card';
        
        card.innerHTML = `
            <div class="list-card-header">
                <h3 class="list-card-title">${list.name}</h3>
                <span class="list-card-badge">${list.category}</span>
            </div>
            <p class="list-card-description">${list.description}</p>
            <div class="list-card-footer">
                <span class="list-card-meta">by ${list.creator}</span>
                <span class="list-card-meta">${list.destinations.length} places</span>
            </div>
        `;
        
        grid.appendChild(card);
    });
}

// Planner List
function renderPlannerList() {
    const plannerList = document.getElementById('plannerList');
    plannerList.innerHTML = '';
    
    if (destinations.length === 0) {
        plannerList.innerHTML = '<p style="text-align: center; color: #666;">No destinations yet. Add some to start planning!</p>';
        return;
    }
    
    destinations.forEach((dest, index) => {
        const item = document.createElement('div');
        item.className = 'planner-item';
        item.draggable = true;
        item.dataset.id = dest.id;
        
        item.innerHTML = `
            <div class="planner-item-number">${index + 1}</div>
            <div class="planner-item-icon">${dest.emoji}</div>
            <div class="planner-item-details">
                <div class="planner-item-name">${dest.name}</div>
                <div class="planner-item-category">${dest.category}</div>
            </div>
            <div class="planner-item-drag">⋮⋮</div>
        `;
        
        // Drag and drop for reordering
        item.addEventListener('dragstart', handleDragStart);
        item.addEventListener('dragover', handleDragOver);
        item.addEventListener('drop', handleDrop);
        item.addEventListener('dragend', handleDragEnd);
        
        plannerList.appendChild(item);
    });
}

function handleDragStart(e) {
    draggedListItem = this;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
}

function handleDragOver(e) {
    if (e.preventDefault) {
        e.preventDefault();
    }
    e.dataTransfer.dropEffect = 'move';
    
    if (this !== draggedListItem) {
        const rect = this.getBoundingClientRect();
        const midpoint = rect.top + rect.height / 2;
        
        if (e.clientY < midpoint) {
            this.parentNode.insertBefore(draggedListItem, this);
        } else {
            this.parentNode.insertBefore(draggedListItem, this.nextSibling);
        }
    }
    
    return false;
}

function handleDrop(e) {
    if (e.stopPropagation) {
        e.stopPropagation();
    }
    return false;
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    
    // Reorder destinations array based on DOM order
    const items = document.querySelectorAll('.planner-item');
    const newOrder = Array.from(items).map(item => item.dataset.id);
    
    destinations = newOrder.map(id => destinations.find(d => d.id === id));
    
    renderPlannerList();
}

// Add CSS for animation
const style = document.createElement('style');
style.textContent = `
    @keyframes dash {
        to {
            stroke-dashoffset: -15;
        }
    }
`;
document.head.appendChild(style);
