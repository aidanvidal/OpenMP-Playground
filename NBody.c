#include <math.h>
#include <SDL2/SDL.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NUM_PARTICLES 20000
#define G 1          // Gravitational constant
#define TIME_STEP 0.05 // Smaller time step for stability
#define SOFTENING 10  // Softening factor to prevent singularities
#define TAU (2.0 * M_PI)
#define SEED 3
#define WIDTH 1200
#define HEIGHT 800

// Barnes-Hut parameters
#define THETA 0.5    // Threshold for approximating distant nodes (0.5 is typical)

// Add zoom variables
double zoomLevel = 1.0;
double zoomSpeed = 0.1; // How quickly zoom changes with scroll

// Fix struct declaration order
struct Particle
{
    double x, y;
    double vx, vy;
    float mass;
    double ax, ay;
};

struct Particle particles[NUM_PARTICLES];

// Quadtree node structure for Barnes-Hut algorithm
struct QuadTreeNode
{
    double x, y;          // Center of mass
    double mass;          // Total mass
    double width;         // Width of this quadrant
    double x_min, y_min;  // Top-left corner
    int particle_index;   // Index of particle (if this is a leaf node with 1 particle)
    int num_particles;    // Number of particles in this node
    struct QuadTreeNode* children[4]; // NW, NE, SW, SE quadrants
};

// Helper function to create a new QuadTreeNode
struct QuadTreeNode* createNode(double x_min, double y_min, double width)
{
    struct QuadTreeNode* node = (struct QuadTreeNode*)malloc(sizeof(struct QuadTreeNode));
    node->x = 0.0;
    node->y = 0.0;
    node->mass = 0.0;
    node->width = width;
    node->x_min = x_min;
    node->y_min = y_min;
    node->particle_index = -1;
    node->num_particles = 0;
    for (int i = 0; i < 4; i++) {
        node->children[i] = NULL;
    }
    return node;
}

// Helper function to free the quadtree
void freeQuadTree(struct QuadTreeNode* node)
{
    if (node == NULL) return;
    
    for (int i = 0; i < 4; i++) {
        if (node->children[i] != NULL) {
            freeQuadTree(node->children[i]);
        }
    }
    
    free(node);
}

// Helper function to determine which quadrant a particle belongs to
int getQuadrant(struct QuadTreeNode* node, double x, double y)
{
    double mid_x = node->x_min + node->width / 2.0;
    double mid_y = node->y_min + node->width / 2.0;
    
    if (x < mid_x) {
        if (y < mid_y) return 0; // NW
        else return 2; // SW
    } else {
        if (y < mid_y) return 1; // NE
        else return 3; // SE
    }
}

// Insert a particle into the quadtree
void insertParticle(struct QuadTreeNode* node, int particle_idx)
{
    // If node doesn't contain any particles yet
    if (node->num_particles == 0) {
        node->particle_index = particle_idx;
        node->x = particles[particle_idx].x;
        node->y = particles[particle_idx].y;
        node->mass = particles[particle_idx].mass;
        node->num_particles = 1;
        return;
    }
    
    // If node contains exactly one particle and is a leaf node
    if (node->num_particles == 1 && node->particle_index != -1) {
        int old_idx = node->particle_index;
        
        // Create children if they don't exist
        double new_width = node->width / 2.0;
        for (int i = 0; i < 4; i++) {
            if (node->children[i] == NULL) {
                double x_offset = (i == 1 || i == 3) ? new_width : 0;
                double y_offset = (i == 2 || i == 3) ? new_width : 0;
                node->children[i] = createNode(node->x_min + x_offset, node->y_min + y_offset, new_width);
            }
        }
        
        // Move the existing particle to the appropriate child
        int q = getQuadrant(node, particles[old_idx].x, particles[old_idx].y);
        insertParticle(node->children[q], old_idx);
        
        // Insert the new particle
        q = getQuadrant(node, particles[particle_idx].x, particles[particle_idx].y);
        insertParticle(node->children[q], particle_idx);
        
        // This node is no longer a leaf
        node->particle_index = -1;
    }
    // If this node already has multiple particles
    else {
        // Insert the new particle into the appropriate child
        int q = getQuadrant(node, particles[particle_idx].x, particles[particle_idx].y);
        if (node->children[q] == NULL) {
            double new_width = node->width / 2.0;
            double x_offset = (q == 1 || q == 3) ? new_width : 0;
            double y_offset = (q == 2 || q == 3) ? new_width : 0;
            node->children[q] = createNode(node->x_min + x_offset, node->y_min + y_offset, new_width);
        }
        insertParticle(node->children[q], particle_idx);
    }
    
    // Update this node's center of mass and total mass
    node->num_particles++;
    double total_mass = node->mass + particles[particle_idx].mass;
    node->x = (node->x * node->mass + particles[particle_idx].x * particles[particle_idx].mass) / total_mass;
    node->y = (node->y * node->mass + particles[particle_idx].y * particles[particle_idx].mass) / total_mass;
    node->mass = total_mass;
}

// Build the quadtree
struct QuadTreeNode* buildQuadTree()
{
    // Find bounds of the simulation
    double min_x = particles[0].x;
    double max_x = particles[0].x;
    double min_y = particles[0].y;
    double max_y = particles[0].y;
    
    for (int i = 1; i < NUM_PARTICLES; i++) {
        if (particles[i].x < min_x) min_x = particles[i].x;
        if (particles[i].x > max_x) max_x = particles[i].x;
        if (particles[i].y < min_y) min_y = particles[i].y;
        if (particles[i].y > max_y) max_y = particles[i].y;
    }
    
    // Add some margin
    double margin = fmax(max_x - min_x, max_y - min_y) * 0.05;
    min_x -= margin;
    max_x += margin;
    min_y -= margin;
    max_y += margin;
    
    // Make it square
    double width = fmax(max_x - min_x, max_y - min_y);
    
    // Create root node
    struct QuadTreeNode* root = createNode(min_x, min_y, width);
    
    // Insert all particles
    for (int i = 0; i < NUM_PARTICLES; i++) {
        insertParticle(root, i);
    }
    
    return root;
}

// Calculate acceleration on a particle from a quadtree node
void calculateForceFromNode(struct QuadTreeNode* node, int particle_idx, double* ax, double* ay)
{
    if (node == NULL) return;
    
    // If this is a leaf with exactly one particle
    if (node->num_particles == 1 && node->particle_index == particle_idx) {
        return; // Skip self-interaction
    }
    
    double dx = node->x - particles[particle_idx].x;
    double dy = node->y - particles[particle_idx].y;
    double dist2 = dx*dx + dy*dy + SOFTENING;
    
    // If this is a leaf with one particle or the node is distant enough
    // (width/distance < theta), use the node's center of mass
    if (node->num_particles == 1 || (node->width*node->width / dist2 < THETA*THETA)) {
        double dist = sqrt(dist2);
        double force = G * node->mass / (dist2 * dist);
        *ax += force * dx;
        *ay += force * dy;
    }
    // Otherwise, recursively check children
    else {
        for (int i = 0; i < 4; i++) {
            if (node->children[i] != NULL) {
                calculateForceFromNode(node->children[i], particle_idx, ax, ay);
            }
        }
    }
}

// Helper function to generate a random point in a unit disc
void rand_disc(double *x, double *y)
{
    double theta = ((double)rand() / RAND_MAX) * TAU;
    double radius = sqrt((double)rand() / RAND_MAX); // Square root for uniform distribution

    *x = radius * cos(theta);
    *y = radius * sin(theta);
}

void initParticles()
{
    // Seed the random number generator
    srand(SEED);

    // Initialize the first particle at the center of the screen to be massive
    particles[0].x = 0.0;
    particles[0].y = 0.0;
    particles[0].vx = 0.0;
    particles[0].vy = 0.0;
    particles[0].mass = 100000.0; // Massive central particle
    particles[0].ax = 0.0;
    particles[0].ay = 0.0;

    for (int i = 1; i < NUM_PARTICLES; i++)
    {
        // Position within a disc
        double pos_x, pos_y;
        rand_disc(&pos_x, &pos_y);

        // Should be more particles near the center and randomly distributed
        double random_factor = ((double)rand() / RAND_MAX);
        particles[i].x = pos_x * random_factor * 1000.0; // Scale to a larger area
        particles[i].y = pos_y * random_factor * 1000.0;

        // Calculate the distance to the central mass
        double distance = sqrt(particles[i].x * particles[i].x + particles[i].y * particles[i].y);
        
        // Calculate orbital velocity for a circular orbit
        // v_orbit = sqrt(G * M / r) with some randomization
        double orbital_speed = sqrt((G * particles[0].mass) / distance);
        
        // For circular orbit, velocity should be perpendicular to position vector
        // Normalize position vector
        double norm_x = particles[i].x / distance;
        double norm_y = particles[i].y / distance;
        
        // Get perpendicular vector (rotate 90 degrees)
        double perp_x = -norm_y;
        double perp_y = norm_x;
        
        // Add some eccentricity (0.7 to 1.3 times circular velocity)
        double eccentricity = 0.7 + ((double)rand() / RAND_MAX) * 0.6;
        
        // Set velocity components
        particles[i].vx = perp_x * orbital_speed * eccentricity;
        particles[i].vy = perp_y * orbital_speed * eccentricity;

        // Small random masses between 0.5 and 1.5
        particles[i].mass = 0.5 + ((float)rand() / RAND_MAX);

        // Initialize accelerations to zero
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
    }
}

// Calculate net forces on all particles using Barnes-Hut algorithm
void update()
{
    // Build the quadtree
    struct QuadTreeNode* quadTree = buildQuadTree();
    
    // Calculate forces on particles using the quadtree
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        double ax_local = 0.0;
        double ay_local = 0.0;
        
        // Calculate force from quadtree
        calculateForceFromNode(quadTree, i, &ax_local, &ay_local);
        
        // Update accelerations for particle i
        particles[i].ax = ax_local;
        particles[i].ay = ay_local;
    }
    
    // Free the quadtree
    freeQuadTree(quadTree);
    
    // Update positions and velocities
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        // Update velocity first (semi-implicit Euler is more stable)
        particles[i].vx += particles[i].ax * TIME_STEP;
        particles[i].vy += particles[i].ay * TIME_STEP;

        // Then update position
        particles[i].x += particles[i].vx * TIME_STEP;
        particles[i].y += particles[i].vy * TIME_STEP;
    }
}

// Map simulation coordinates to screen coordinates
void mapToScreen(double x, double y, int *screenX, int *screenY)
{
    // Map from simulation space to screen space
    double scale = (WIDTH < HEIGHT) ? WIDTH / 200.0 : HEIGHT / 200.0;
    scale *= zoomLevel; // Apply zoom
    *screenX = (int)(WIDTH / 2 + x * scale);
    *screenY = (int)(HEIGHT / 2 - y * scale); // Y is flipped in screen coordinates
}

// Draw particles to the screen
void renderParticles(SDL_Renderer *renderer)
{
    // Clear the screen
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Can't use threads here as SDL is not thread-safe
    // Draw each particle
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        // Map particle position to screen coordinates
        int screenX, screenY;
        mapToScreen(particles[i].x, particles[i].y, &screenX, &screenY);

        // Determine radius based on mass and on zoom level
        int radius = 1 + (int)(sqrt(particles[i].mass)) * zoomLevel;

        // Set color based on speed using a gradient
        double speed = sqrt(particles[i].vx * particles[i].vx + particles[i].vy * particles[i].vy);
        double max_speed = 70.0; // Adjust based on expected maximum speed
        double normalized_speed = fmin(speed / max_speed, 1.0);

        // Gradient: Blue (slow) -> Red (medium) -> Yellow (fast)
        uint8_t r, g, b;
        if (normalized_speed < 0.5)
        {
            // Blue to Red gradient
            r = (uint8_t)(255 * (normalized_speed * 2.0));
            g = 0;
            b = (uint8_t)(255 * (1.0 - normalized_speed * 2.0));
        }
        else
        {
            // Red to Yellow gradient
            r = 255;
            g = (uint8_t)(255 * ((normalized_speed - 0.5) * 2.0));
            b = 0;
        }
        if(i == 0) // Central mass
        {
            r = 0; g = 0; b = 0; // Black for the central mass
        }
        SDL_SetRenderDrawColor(renderer, r, g, b, 255);

        // Draw the particle as a filled circle
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx * dx + dy * dy <= radius * radius)
                {
                    SDL_RenderDrawPoint(renderer, screenX + dx, screenY + dy);
                }
            }
        }
    }

    // Present the rendered frame
    SDL_RenderPresent(renderer);
}

int main(int argc, char *argv[])
{
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Create window
    SDL_Window *window = SDL_CreateWindow("N-Body Simulation",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          WIDTH, HEIGHT,
                                          SDL_WINDOW_SHOWN);
    if (!window)
    {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Create renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Initialize particles
    initParticles();

    // Main loop
    int quit = 0;
    SDL_Event e;

    while (!quit)
    {
        // Handle events
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
            {
                quit = 1;
            }
            else if (e.type == SDL_MOUSEWHEEL)
            {
                // Zoom in/out with mouse wheel
                if (e.wheel.y > 0) // Scroll up
                {
                    zoomLevel *= (1.0 + zoomSpeed);
                }
                else if (e.wheel.y < 0) // Scroll down
                {
                    zoomLevel *= (1.0 - zoomSpeed);
                }

                // Set limits to prevent extreme zoom levels
                if (zoomLevel < 0.1)
                    zoomLevel = 0.1;
                if (zoomLevel > 10.0)
                    zoomLevel = 10.0;
            }
        }

        // Calculate forces and update positions
        update();

        // Render particles
        renderParticles(renderer);

        // Cap frame rate to roughly 120 FPS
        SDL_Delay(8);
    }

    // Clean up
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
