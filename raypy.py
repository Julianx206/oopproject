import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import math

class Vector:
    """A 2D vector class for vector operations."""
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)
    
    def perpendicular(self):
        """Returns a perpendicular vector (rotate 90 degrees counterclockwise)"""
        return Vector(-self.y, self.x)
    
    def as_tuple(self):
        return (self.x, self.y)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"


class Ray:
    """Represents a light ray with a position and direction."""
    
    def __init__(self, position, direction, wavelength=550, intensity=1.0):
        """
        Initialize a ray.
        
        Args:
            position (Vector): The starting position of the ray
            direction (Vector): The direction of the ray (will be normalized)
            wavelength (float): Wavelength in nm (default is 550nm, green light)
            intensity (float): Light intensity from 0 to 1
        """
        self.position = position
        self.direction = direction.normalize()
        self.wavelength = wavelength
        self.intensity = intensity
        self.path = [(position.x, position.y)]
        self.active = True
    
    def propagate(self, distance):
        """Move the ray forward by the specified distance."""
        if self.active:
            self.position = self.position + self.direction * distance
            self.path.append((self.position.x, self.position.y))
    
    def get_color(self):
        """Get RGB color based on wavelength."""
        # Simplified conversion from wavelength to RGB
        # This is a very basic approximation
        if 380 <= self.wavelength < 440:
            r, g, b = 0.3, 0.0, 0.7  # Violet
        elif 440 <= self.wavelength < 490:
            r, g, b = 0.0, 0.0, 1.0  # Blue
        elif 490 <= self.wavelength < 510:
            r, g, b = 0.0, 1.0, 1.0  # Cyan
        elif 510 <= self.wavelength < 580:
            r, g, b = 0.0, 1.0, 0.0  # Green
        elif 580 <= self.wavelength < 640:
            r, g, b = 1.0, 1.0, 0.0  # Yellow
        elif 640 <= self.wavelength < 700:
            r, g, b = 1.0, 0.0, 0.0  # Red
        else:
            r, g, b = 0.5, 0.5, 0.5  # Outside visible spectrum
        
        return (r, g, b, self.intensity)


class OpticalElement:
    """Base class for all optical elements."""
    
    def __init__(self, position=Vector(0, 0)):
        self.position = position
    
    def interact(self, ray):
        """Interact with a ray. To be implemented by subclasses."""
        pass
    
    def draw(self, ax):
        """Draw the optical element. To be implemented by subclasses."""
        pass


class Mirror(OpticalElement):
    """A linear mirror that reflects rays."""
    
    def __init__(self, start_point, end_point):
        """
        Initialize a mirror segment.
        
        Args:
            start_point (Vector): Start point of the mirror
            end_point (Vector): End point of the mirror
        """
        super().__init__((start_point + end_point) * 0.5)  # Center position
        self.start_point = start_point
        self.end_point = end_point
        self.direction = (end_point - start_point).normalize()
        self.normal = self.direction.perpendicular()
    
    def interact(self, ray):
        """Reflect the ray if it intersects with the mirror."""
        # Vector from ray position to mirror start
        v1 = self.start_point - ray.position
        v2 = self.end_point - ray.position
        
        # Check if ray intersects with mirror segment
        cross1 = v1.x * ray.direction.y - v1.y * ray.direction.x
        cross2 = v2.x * ray.direction.y - v2.y * ray.direction.x
        
        if cross1 * cross2 <= 0:
            # Line from start to end
            mirror_dir = self.end_point - self.start_point
            mirror_len = mirror_dir.magnitude()
            mirror_dir = mirror_dir * (1/mirror_len)
            
            # Perpendicular distance from ray to mirror line
            perp_dist = (self.start_point - ray.position).dot(mirror_dir.perpendicular())
            
            # Ray direction dotted with perpendicular
            ray_dot_perp = ray.direction.dot(mirror_dir.perpendicular())
            
            # If ray is parallel to mirror or pointing away
            if abs(ray_dot_perp) < 1e-10:
                return
            
            # Distance along ray to intersection
            t = perp_dist / ray_dot_perp
            
            # If intersection is behind the ray
            if t <= 0:
                return
            
            # Intersection point
            intersection = ray.position + ray.direction * t
            
            # Check if intersection is on mirror segment
            mirror_t = (intersection - self.start_point).dot(mirror_dir) / mirror_len
            if 0 <= mirror_t <= 1:
                # Move ray to intersection
                ray.position = intersection
                ray.path.append((ray.position.x, ray.position.y))
                
                # Reflect ray direction: d' = d - 2(d·n)n
                dot_product = ray.direction.dot(self.normal)
                ray.direction = ray.direction - self.normal * (2 * dot_product)
    
    def draw(self, ax):
        """Draw the mirror on the plot."""
        ax.plot([self.start_point.x, self.end_point.x], 
                [self.start_point.y, self.end_point.y], 
                'k-', linewidth=2)


class GlassBlock(OpticalElement):
    """A rectangular glass block that refracts light."""
    
    def __init__(self, position, width, height, refractive_index=1.5):
        """
        Initialize a glass block.
        
        Args:
            position (Vector): The center position of the block
            width (float): Width of the block
            height (float): Height of the block
            refractive_index (float): Refractive index of the glass
        """
        super().__init__(position)
        self.width = width
        self.height = height
        self.n = refractive_index  # Refractive index
        
        # Calculate the four corners
        self.corners = [
            Vector(position.x - width/2, position.y - height/2),  # Bottom left
            Vector(position.x + width/2, position.y - height/2),  # Bottom right
            Vector(position.x + width/2, position.y + height/2),  # Top right
            Vector(position.x - width/2, position.y + height/2)   # Top left
        ]
        
        # Four sides as line segments
        self.sides = [
            (self.corners[0], self.corners[1]),  # Bottom
            (self.corners[1], self.corners[2]),  # Right
            (self.corners[2], self.corners[3]),  # Top
            (self.corners[3], self.corners[0])   # Left
        ]
        
        # Normals pointing outward for each side
        self.normals = [
            Vector(0, -1),  # Bottom
            Vector(1, 0),   # Right
            Vector(0, 1),   # Top
            Vector(-1, 0)   # Left
        ]
    
    def is_inside(self, point):
        """Check if a point is inside the glass block."""
        return (self.position.x - self.width/2 <= point.x <= self.position.x + self.width/2 and
                self.position.y - self.height/2 <= point.y <= self.position.y + self.height/2)
    
    def _find_intersection(self, ray):
        """Find the first intersection of the ray with any side of the block."""
        closest_t = float('inf')
        closest_side = -1
        
        for i, (start, end) in enumerate(self.sides):
            # Vector representing the side
            side_vec = end - start
            side_len = side_vec.magnitude()
            side_dir = side_vec * (1/side_len)
            
            # Perpendicular distance from ray to side line
            perp_dist = (start - ray.position).dot(side_dir.perpendicular())
            
            # Ray direction dotted with perpendicular
            ray_dot_perp = ray.direction.dot(side_dir.perpendicular())
            
            # If ray is parallel to side
            if abs(ray_dot_perp) < 1e-10:
                continue
            
            # Distance along ray to intersection
            t = perp_dist / ray_dot_perp
            
            # If intersection is behind the ray
            if t <= 0:
                continue
            
            # Intersection point
            intersection = ray.position + ray.direction * t
            
            # Check if intersection is on side segment
            side_t = (intersection - start).dot(side_dir)
            if 0 <= side_t <= side_len:
                if t < closest_t:
                    closest_t = t
                    closest_side = i
        
        if closest_side != -1:
            return closest_t, closest_side
        return None, None
    
    def _refract(self, incident, normal, n1, n2):
        """
        Calculate the refracted ray direction using Snell's law.
        
        Args:
            incident (Vector): Incident ray direction (normalized)
            normal (Vector): Surface normal (normalized)
            n1 (float): Refractive index of medium 1
            n2 (float): Refractive index of medium 2
            
        Returns:
            Vector: Refracted ray direction or None if total internal reflection
        """
        # Make sure normal points against incident direction
        if incident.dot(normal) > 0:
            normal = normal * -1
        
        # Cosine of angle between incident and normal
        cos_theta1 = -incident.dot(normal)
        
        # Compute sin^2(theta2) using Snell's law
        sin_theta2_sq = (n1 / n2)**2 * (1 - cos_theta1**2)
        
        # Check for total internal reflection
        if sin_theta2_sq > 1:
            # Total internal reflection: reflect the ray
            return incident + normal * (2 * cos_theta1)
        
        # Compute refracted direction
        cos_theta2 = math.sqrt(1 - sin_theta2_sq)
        return (incident * (n1 / n2)) + normal * ((n1 / n2) * cos_theta1 - cos_theta2)
    
    def interact(self, ray):
        """Handle ray interaction with the glass block."""
        is_inside = self.is_inside(ray.position)
        n1 = 1.0 if not is_inside else self.n  # Air or glass
        n2 = self.n if not is_inside else 1.0  # Glass or air
        
        # Find intersection with the block
        t, side_idx = self._find_intersection(ray)
        
        if t is not None and side_idx is not None:
            # Move ray to intersection point
            intersection = ray.position + ray.direction * t
            ray.position = intersection
            ray.path.append((ray.position.x, ray.position.y))
            
            # Get normal at intersection point
            normal = self.normals[side_idx]
            
            # Refract the ray
            ray.direction = self._refract(ray.direction, normal, n1, n2).normalize()
    
    def draw(self, ax):
        """Draw the glass block on the plot."""
        # Create polygon
        corners_xy = [(corner.x, corner.y) for corner in self.corners]
        polygon = Polygon(corners_xy, alpha=0.2, facecolor='lightblue', edgecolor='blue')
        ax.add_patch(polygon)


class Lens(OpticalElement):
    """A simple lens represented as a line segment with refractive power."""
    
    def __init__(self, position, width, focal_length, orientation=0):
        """
        Initialize a lens.
        
        Args:
            position (Vector): Center position of the lens
            width (float): Width/diameter of the lens
            focal_length (float): Focal length of the lens (positive for converging, negative for diverging)
            orientation (float): Orientation in radians (default is vertical lens)
        """
        super().__init__(position)
        self.width = width
        self.focal_length = focal_length
        self.orientation = orientation
        
        # Direction along the lens
        self.direction = Vector(math.sin(orientation), math.cos(orientation)).normalize()
        
        # Perpendicular to the lens (optical axis)
        self.normal = Vector(math.cos(orientation), -math.sin(orientation)).normalize()
        
        # Endpoints of the lens
        self.start_point = position - self.direction * (width / 2)
        self.end_point = position + self.direction * (width / 2)
    
    def interact(self, ray):
        """Refract the ray through the lens using the thin lens approximation."""
        # Vector from ray position to lens start
        v1 = self.start_point - ray.position
        v2 = self.end_point - ray.position
        
        # Check if ray intersects with lens segment
        cross1 = v1.x * ray.direction.y - v1.y * ray.direction.x
        cross2 = v2.x * ray.direction.y - v2.y * ray.direction.x
        
        if cross1 * cross2 <= 0:
            # Line from start to end
            lens_dir = self.end_point - self.start_point
            lens_len = lens_dir.magnitude()
            lens_dir = lens_dir * (1/lens_len)
            
            # Perpendicular distance from ray to lens line
            perp_dist = (self.start_point - ray.position).dot(lens_dir.perpendicular())
            
            # Ray direction dotted with perpendicular
            ray_dot_perp = ray.direction.dot(lens_dir.perpendicular())
            
            # If ray is parallel to lens
            if abs(ray_dot_perp) < 1e-10:
                return
            
            # Distance along ray to intersection
            t = perp_dist / ray_dot_perp
            
            # If intersection is behind the ray
            if t <= 0:
                return
            
            # Intersection point
            intersection = ray.position + ray.direction * t
            
            # Check if intersection is on lens segment
            lens_t = (intersection - self.start_point).dot(lens_dir) / lens_len
            if 0 <= lens_t <= 1:
                # Move ray to intersection
                ray.position = intersection
                ray.path.append((ray.position.x, ray.position.y))
                
                # For thin lens: calculate the height (distance from optical axis)
                height = (intersection - self.position).dot(self.direction)
                
                # Calculate angle deviation based on height and focal length
                # For a thin lens: tan(θ) = -h/f where h is height, f is focal length
                angle_deviation = -height / self.focal_length
                
                # Create a rotation matrix for the angle deviation
                cos_a = math.cos(angle_deviation)
                sin_a = math.sin(angle_deviation)
                
                # Convert ray direction to coordinates in the lens's reference frame
                dot_normal = ray.direction.dot(self.normal)
                dot_dir = ray.direction.dot(self.direction)
                
                # Apply the thin lens transformation (refraction)
                new_normal_component = dot_normal  # Stays the same in thin lens approximation
                new_dir_component = cos_a * dot_dir + sin_a * dot_normal
                
                # Convert back to global coordinates
                ray.direction = (self.normal * new_normal_component + 
                                 self.direction * new_dir_component).normalize()
    
    def draw(self, ax):
        """Draw the lens on the plot."""
        line = plt.Line2D([self.start_point.x, self.end_point.x], 
                          [self.start_point.y, self.end_point.y], 
                          color='purple', linewidth=3)
        ax.add_line(line)
        
        # Draw small perpendicular lines to indicate lens type
        if self.focal_length > 0:  # Converging lens (thicker in the middle)
            # Draw two arcs to represent a converging lens
            center_perp = 0.1 * self.width
            for point, direction in [(self.start_point, -1), (self.end_point, 1)]:
                perp_start = point + self.normal * center_perp
                perp_end = point - self.normal * center_perp
                ax.plot([perp_start.x, perp_end.x], [perp_start.y, perp_end.y], 
                        color='purple', linewidth=1)
        else:  # Diverging lens (thinner in the middle)
            # Draw two arcs to represent a diverging lens
            edge_perp = 0.1 * self.width
            mid_point = (self.start_point + self.end_point) * 0.5
            
            # Start point edge
            perp_start = self.start_point + self.normal * edge_perp
            perp_end = self.start_point - self.normal * edge_perp
            ax.plot([perp_start.x, perp_end.x], [perp_start.y, perp_end.y], 
                    color='purple', linewidth=1)
            
            # End point edge
            perp_start = self.end_point + self.normal * edge_perp
            perp_end = self.end_point - self.normal * edge_perp
            ax.plot([perp_start.x, perp_end.x], [perp_start.y, perp_end.y], 
                    color='purple', linewidth=1)


class LightSource:
    """A source that emits rays of light."""
    
    def __init__(self, position, num_rays=1, angle_range=(0, 2*math.pi), wavelength=550):
        """
        Initialize a light source.
        
        Args:
            position (Vector): Position of the light source
            num_rays (int): Number of rays to emit
            angle_range (tuple): Range of angles to emit rays in (start, end) in radians
            wavelength (float or tuple): Wavelength in nm or range of wavelengths (min, max)
        """
        self.position = position
        self.num_rays = num_rays
        self.angle_range = angle_range
        self.wavelength = wavelength
    
    def emit_rays(self):
        """Emit rays from the light source."""
        rays = []
        
        if self.num_rays == 1:
            # Single ray in the middle of the angle range
            angle = (self.angle_range[0] + self.angle_range[1]) / 2
            direction = Vector(math.cos(angle), math.sin(angle))
            
            if isinstance(self.wavelength, tuple):
                wl = (self.wavelength[0] + self.wavelength[1]) / 2
            else:
                wl = self.wavelength
                
            rays.append(Ray(Vector(self.position.x, self.position.y), direction, wl))
        else:
            # Multiple rays evenly distributed across the angle range
            angle_step = (self.angle_range[1] - self.angle_range[0]) / max(1, self.num_rays - 1)
            
            for i in range(self.num_rays):
                angle = self.angle_range[0] + i * angle_step
                direction = Vector(math.cos(angle), math.sin(angle))
                
                if isinstance(self.wavelength, tuple):
                    # Distribute wavelengths evenly across the range
                    t = i / max(1, self.num_rays - 1)
                    wl = self.wavelength[0] + t * (self.wavelength[1] - self.wavelength[0])
                else:
                    wl = self.wavelength
                
                rays.append(Ray(Vector(self.position.x, self.position.y), direction, wl))
        
        return rays
    
    def draw(self, ax):
        """Draw the light source on the plot."""
        circle = plt.Circle((self.position.x, self.position.y), 0.1, color='yellow', fill=True)
        ax.add_patch(circle)


class Scene:
    """A scene containing optical elements and light sources."""
    
    def __init__(self, size=(10, 10)):
        """
        Initialize a scene.
        
        Args:
            size (tuple): Size of the scene (width, height)
        """
        self.size = size
        self.optical_elements = []
        self.light_sources = []
        self.rays = []
        
    def add_optical_element(self, element):
        """Add an optical element to the scene."""
        self.optical_elements.append(element)
        
    def add_light_source(self, source):
        """Add a light source to the scene."""
        self.light_sources.append(source)
    
    def emit_all_rays(self):
        """Emit rays from all light sources."""
        self.rays = []
        for source in self.light_sources:
            self.rays.extend(source.emit_rays())
    
    def trace_rays(self, max_steps=100, step_size=0.1):
        """
        Trace all rays through the scene.
        
        Args:
            max_steps (int): Maximum number of propagation steps
            step_size (float): Size of each propagation step
        """
        # Emit rays if not already done
        if not self.rays:
            self.emit_all_rays()
        
        # Trace each ray
        for _ in range(max_steps):
            # Propagate all active rays
            for ray in self.rays:
                if ray.active:
                    ray.propagate(step_size)
                    
                    # Check if ray is out of bounds
                    if (ray.position.x < -self.size[0]/2 or ray.position.x > self.size[0]/2 or
                        ray.position.y < -self.size[1]/2 or ray.position.y > self.size[1]/2):
                        ray.active = False
                        continue
                    
                    # Check for interactions with optical elements
                    for element in self.optical_elements:
                        element.interact(ray)
    
    def draw(self, ax=None, show_elements=True, show_rays=True, show_sources=True):
        """
        Draw the scene.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to draw on (creates new figure if None)
            show_elements (bool): Whether to draw optical elements
            show_rays (bool): Whether to draw rays
            show_sources (bool): Whether to draw light sources
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set axis limits
        w, h = self.size
        ax.set_xlim(-w/2, w/2)
        ax.set_ylim(-h/2, h/2)
        ax.set_aspect('equal')
        
        # Draw optical elements
        if show_elements:
            for element in self.optical_elements:
                element.draw(ax)
        
        # Draw light sources
        if show_sources:
            for source in self.light_sources:
                source.draw(ax)
        
        # Draw rays
        if show_rays:
            for ray in self.rays:
                path = np.array(ray.path)
                if len(path) > 1:
                    ax.plot(path[:, 0], path[:, 1], color=ray.get_color(), linewidth=1)
        
        return ax


class Simulator:
    """High-level simulator class to manage scenes and simulations."""
    
    def __init__(self):
        """Initialize the simulator."""
        self.scene = None
    
    def create_scene(self, size=(10, 10)):
        """Create a new scene."""
        self.scene = Scene(size)
        return self.scene
    
    def simulate(self, max_steps=100, step_size=0.1):
        """Run the simulation on the current scene."""
        if self.scene:
            self.scene.emit_all_rays()
            self.scene.trace_rays(max_steps, step_size)
    
    def display(self, figsize=(10, 10)):
        """Display the current scene."""
        if self.scene:
            fig, ax = plt.subplots(figsize=figsize)
            self.scene.draw(ax)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title('Ray Optics Simulation')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            plt.show()
    
    def animate(self, max_steps=100, step_size=0.1, interval=50):
        """
        Create an animation of the ray tracing.
        
        Args:
            max_steps (int): Maximum number of simulation steps
            step_size (float): Size of each simulation step
            interval (int): Interval between frames in milliseconds
        
        Returns:
            matplotlib.animation.FuncAnimation: Animation object
        """
        if not self.scene:
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        w, h = self.scene.size
        ax.set_xlim(-w/2, w/2)
        ax.set_ylim(-h/2, h/2)
        ax.set_aspect('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title('Ray Optics Simulation')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Draw optical elements and sources (static)
        for element in self.scene.optical_elements:
            element.draw(ax)
        
        for source in self.scene.light_sources:
            source.draw(ax)
        
        # Emit rays
        self.scene.emit_all_rays()
        
        # Lines for each ray
        lines = []
        colors = []
        for ray in self.scene.rays:
            line, = ax.plot([], [], linewidth=1)
            lines.append(line)
            colors.append(ray.get_color())
        
        def init():
            """Initialize animation."""
            for line in lines:
                line.set_data([], [])
            return lines
        
        def animate(i):
            """Animation function."""
            # Only run simulation steps if there are active rays
            if any(ray.active for ray in self.scene.rays):
                # Propagate rays for one step
                for ray in self.scene.rays:
                    if ray.active:
                        ray.propagate(step_size)
                        
                        # Check if ray is out of bounds
                        if (ray.position.x < -w/2 or ray.position.x > w/2 or
                            ray.position.y < -h/2 or ray.position.y > h/2):
                            ray.active = False
                            continue
                        
                        # Check for interactions with optical elements
                        for element in self.scene.optical_elements:
                            element.interact(ray)
            
            # Update lines
            for j, (ray, line) in enumerate(zip(self.scene.rays, lines)):
                path = np.array(ray.path)
                if len(path) > 1:
                    line.set_data(path[:, 0], path[:, 1])
                    line.set_color(colors[j])
            
            return lines
        
        ani = animation.FuncAnimation(
            fig, animate, frames=max_steps, init_func=init, 
            interval=interval, blit=True)
        
        plt.close()  # Prevent display of static figure
        return ani


# Helper functions to create common optical setups
def create_mirror_system(simulator):
    """Create a scene with multiple mirrors."""
    scene = simulator.create_scene()
    
    # Add mirrors
    scene.add_optical_element(Mirror(Vector(-4, -2), Vector(-2, 2)))
    scene.add_optical_element(Mirror(Vector(-2, 2), Vector(2, 2)))
    scene.add_optical_element(Mirror(Vector(2, 2), Vector(4, -2)))
    
    # Add light source
    scene.add_light_source(LightSource(Vector(-3, -3), num_rays=5, angle_range=(0, math.pi/2)))
    
    return scene

def create_lens_system(simulator):
    """Create a scene with converging and diverging lenses."""
    scene = simulator.create_scene()
    
    # Add lenses
    scene.add_optical_element(Lens(Vector(-2, 0), 2, 3))        # Converging lens
    scene.add_optical_element(Lens(Vector(2, 0), 2, -4))        # Diverging lens
    
    # Add light source
    scene.add_light_source(LightSource(Vector(-4, 0), num_rays=7, 
                                      angle_range=(-math.pi/8, math.pi/8)))
    
    return scene

def create_prism_system(simulator):
    """Create a scene with a prism that demonstrates dispersion."""
    scene = simulator.create_scene()
    
    # Create a triangular prism using a custom polygon
    class Prism(OpticalElement):
        """A triangular prism that demonstrates dispersion."""
        
        def __init__(self, position, size, orientation=0, refractive_index=1.5, dispersion=0.02):
            """
            Initialize a triangular prism.
            
            Args:
                position (Vector): Position of the prism center
                size (float): Size of the prism
                orientation (float): Orientation in radians
                refractive_index (float): Base refractive index (at 550nm)
                dispersion (float): Dispersion coefficient
            """
            super().__init__(position)
            self.size = size
            self.orientation = orientation
            self.base_n = refractive_index
            self.dispersion = dispersion
            
            # Create the vertices of an equilateral triangle
            self.vertices = []
            for i in range(3):
                angle = orientation + i * (2 * math.pi / 3)
                x = position.x + size * math.cos(angle)
                y = position.y + size * math.sin(angle)
                self.vertices.append(Vector(x, y))
            
            # Create sides as line segments
            self.sides = []
            self.normals = []
            for i in range(3):
                start = self.vertices[i]
                end = self.vertices[(i + 1) % 3]
                self.sides.append((start, end))
                
                # Calculate outward-pointing normal
                direction = (end - start).normalize()
                normal = direction.perpendicular() * -1  # Outward normal
                self.normals.append(normal)
        
        def get_refractive_index(self, wavelength):
            """
            Get the refractive index for a specific wavelength.
            
            Implements a simple Cauchy's equation n(λ) = A + B/λ²
            """
            # Convert wavelength from nm to μm for the formula
            wl_um = wavelength / 1000
            
            # Simplified dispersion model
            return self.base_n + self.dispersion * (0.55**2 - wl_um**2)
        
        def is_inside(self, point):
            """Check if a point is inside the prism using winding number algorithm."""
            winding_number = 0
            for i in range(3):
                p1 = self.vertices[i]
                p2 = self.vertices[(i + 1) % 3]
                
                if p1.y <= point.y:
                    if p2.y > point.y and (p2 - p1).perpendicular().dot(point - p1) > 0:
                        winding_number += 1
                else:
                    if p2.y <= point.y and (p2 - p1).perpendicular().dot(point - p1) < 0:
                        winding_number -= 1
            
            return winding_number != 0
        
        def _find_intersection(self, ray):
            """Find the first intersection of the ray with any side of the prism."""
            closest_t = float('inf')
            closest_side = -1
            
            for i, (start, end) in enumerate(self.sides):
                # Vector representing the side
                side_vec = end - start
                side_len = side_vec.magnitude()
                side_dir = side_vec * (1/side_len)
                
                # Perpendicular distance from ray to side line
                perp_dist = (start - ray.position).dot(side_dir.perpendicular())
                
                # Ray direction dotted with perpendicular
                ray_dot_perp = ray.direction.dot(side_dir.perpendicular())
                
                # If ray is parallel to side
                if abs(ray_dot_perp) < 1e-10:
                    continue
                
                # Distance along ray to intersection
                t = perp_dist / ray_dot_perp
                
                # If intersection is behind the ray
                if t <= 0:
                    continue
                
                # Intersection point
                intersection = ray.position + ray.direction * t
                
                # Check if intersection is on side segment
                side_t = (intersection - start).dot(side_dir) / side_len
                if 0 <= side_t <= 1:
                    if t < closest_t:
                        closest_t = t
                        closest_side = i
            
            if closest_side != -1:
                return closest_t, closest_side
            return None, None
        
        def _refract(self, incident, normal, n1, n2):
            """
            Calculate the refracted ray direction using Snell's law.
            
            Args:
                incident (Vector): Incident ray direction (normalized)
                normal (Vector): Surface normal (normalized)
                n1 (float): Refractive index of medium 1
                n2 (float): Refractive index of medium 2
                
            Returns:
                Vector: Refracted ray direction or None if total internal reflection
            """
            # Make sure normal points against incident direction
            if incident.dot(normal) > 0:
                normal = normal * -1
            
            # Cosine of angle between incident and normal
            cos_theta1 = -incident.dot(normal)
            
            # Compute sin^2(theta2) using Snell's law
            sin_theta2_sq = (n1 / n2)**2 * (1 - cos_theta1**2)
            
            # Check for total internal reflection
            if sin_theta2_sq > 1:
                # Total internal reflection: reflect the ray
                return incident + normal * (2 * cos_theta1)
            
            # Compute refracted direction
            cos_theta2 = math.sqrt(1 - sin_theta2_sq)
            return (incident * (n1 / n2)) + normal * ((n1 / n2) * cos_theta1 - cos_theta2)
        
        def interact(self, ray):
            """Handle ray interaction with the prism."""
            is_inside = self.is_inside(ray.position)
            
            # Get refractive index based on wavelength
            n_prism = self.get_refractive_index(ray.wavelength)
            
            n1 = 1.0 if not is_inside else n_prism  # Air or prism
            n2 = n_prism if not is_inside else 1.0  # Prism or air
            
            # Find intersection with the prism
            t, side_idx = self._find_intersection(ray)
            
            if t is not None and side_idx is not None:
                # Move ray to intersection point
                intersection = ray.position + ray.direction * t
                ray.position = intersection
                ray.path.append((ray.position.x, ray.position.y))
                
                # Get normal at intersection point
                normal = self.normals[side_idx]
                
                # Refract or reflect the ray
                ray.direction = self._refract(ray.direction, normal, n1, n2).normalize()
        
        def draw(self, ax):
            """Draw the prism on the plot."""
            # Create polygon
            vertices_xy = [(vertex.x, vertex.y) for vertex in self.vertices]
            polygon = Polygon(vertices_xy, alpha=0.2, facecolor='lightcyan', edgecolor='lightblue')
            ax.add_patch(polygon)
    
    # Add a prism to the scene
    prism = Prism(Vector(0, 0), 3, orientation=0)
    scene.add_optical_element(prism)
    
    # Add light source with white light (visible spectrum)
    scene.add_light_source(LightSource(Vector(-4, 0), num_rays=7, 
                                       angle_range=(-math.pi/20, math.pi/20),
                                       wavelength=(380, 700)))  # Visible spectrum
    
    return scene

def create_rainbow_system(simulator):
    """Create a scene demonstrating rainbow formation with water droplets."""
    scene = simulator.create_scene(size=(16, 12))
    
    # Create water droplet class
    class WaterDroplet(OpticalElement):
        """A circular water droplet that refracts and reflects light."""
        
        def __init__(self, position, radius, refractive_index=1.33, dispersion=0.01):
            """
            Initialize a water droplet.
            
            Args:
                position (Vector): Center of the droplet
                radius (float): Radius of the droplet
                refractive_index (float): Base refractive index of water
                dispersion (float): Dispersion coefficient
            """
            super().__init__(position)
            self.radius = radius
            self.base_n = refractive_index
            self.dispersion = dispersion
        
        def get_refractive_index(self, wavelength):
            """Get the refractive index for a specific wavelength."""
            # Convert wavelength from nm to μm for the formula
            wl_um = wavelength / 1000
            
            # Simplified dispersion model
            return self.base_n + self.dispersion * (0.55**2 - wl_um**2)
        
        def _find_intersection(self, ray):
            """Find the intersection of a ray with the droplet."""
            # Vector from center to ray position
            oc = ray.position - self.position
            
            # Quadratic formula coefficients for sphere intersection
            a = ray.direction.dot(ray.direction)
            b = 2.0 * oc.dot(ray.direction)
            c = oc.dot(oc) - self.radius**2
            
            discriminant = b**2 - 4*a*c
            
            if discriminant < 0:
                return None, None  # No intersection
            
            # Find closest intersection
            t = (-b - math.sqrt(discriminant)) / (2.0 * a)
            
            if t <= 0:
                # Try the other solution
                t = (-b + math.sqrt(discriminant)) / (2.0 * a)
                if t <= 0:
                    return None, None  # Both behind the ray
            
            # Calculate the intersection point and normal
            intersection = ray.position + ray.direction * t
            normal = (intersection - self.position).normalize()
            
            return t, normal
        
        def is_inside(self, point):
            """Check if a point is inside the droplet."""
            return (point - self.position).magnitude() < self.radius
        
        def _refract(self, incident, normal, n1, n2):
            """Calculate refracted ray direction using Snell's law."""
            # Make sure normal points against incident direction
            if incident.dot(normal) > 0:
                normal = normal * -1
            
            # Cosine of angle between incident and normal
            cos_theta1 = -incident.dot(normal)
            
            # Compute sin^2(theta2) using Snell's law
            sin_theta2_sq = (n1 / n2)**2 * (1 - cos_theta1**2)
            
            # Check for total internal reflection
            if sin_theta2_sq > 1:
                # Total internal reflection: reflect the ray
                return incident + normal * (2 * cos_theta1)
            
            # Compute refracted direction
            cos_theta2 = math.sqrt(1 - sin_theta2_sq)
            return (incident * (n1 / n2)) + normal * ((n1 / n2) * cos_theta1 - cos_theta2)
        
        def interact(self, ray):
            """Handle ray interaction with the water droplet."""
            is_inside = self.is_inside(ray.position)
            
            # Get refractive index based on wavelength
            n_water = self.get_refractive_index(ray.wavelength)
            
            n1 = 1.0 if not is_inside else n_water  # Air or water
            n2 = n_water if not is_inside else 1.0  # Water or air
            
            # Find intersection with the droplet
            t, normal = self._find_intersection(ray)
            
            if t is not None and normal is not None:
                # Move ray to intersection point
                intersection = ray.position + ray.direction * t
                ray.position = intersection
                ray.path.append((ray.position.x, ray.position.y))
                
                # Refract or reflect the ray
                ray.direction = self._refract(ray.direction, normal, n1, n2).normalize()
        
        def draw(self, ax):
            """Draw the water droplet on the plot."""
            circle = plt.Circle(
                (self.position.x, self.position.y), 
                self.radius, 
                alpha=0.2, 
                facecolor='lightblue', 
                edgecolor='blue'
            )
            ax.add_patch(circle)
    
    # Add several water droplets to create rain
    for i in range(10):
        x = -2 + i * 0.5
        y = 2 - i * 0.4
        droplet = WaterDroplet(Vector(x, y), 0.3)
        scene.add_optical_element(droplet)
    
    # Add sunlight (parallel rays with different wavelengths)
    scene.add_light_source(LightSource(
        Vector(-6, 4), 
        num_rays=15, 
        angle_range=(-math.pi/10, math.pi/10),
        wavelength=(380, 700)  # Visible spectrum
    ))
    
    return scene

def create_microscope_system(simulator):
    """Create a scene simulating a simple microscope."""
    scene = simulator.create_scene(size=(20, 10))
    
    # Add objective lens (strong converging lens)
    objective = Lens(Vector(-2, 0), 2, 2)  # Short focal length
    scene.add_optical_element(objective)
    
    # Add eyepiece lens (weaker converging lens)
    eyepiece = Lens(Vector(4, 0), 3, 4)  # Longer focal length
    scene.add_optical_element(eyepiece)
    
    # Add specimen (represented by a light source)
    specimen = LightSource(Vector(-4, 0), num_rays=5, angle_range=(-math.pi/2, math.pi/2))
    scene.add_light_source(specimen)
    
    return scene

def create_telescope_system(simulator):
    """Create a scene simulating a simple telescope."""
    scene = simulator.create_scene(size=(30, 10))
    
    # Add objective lens (weaker converging lens with larger diameter)
    objective = Lens(Vector(-10, 0), 4, 10)  # Long focal length
    scene.add_optical_element(objective)
    
    # Add eyepiece lens (stronger converging lens)
    eyepiece = Lens(Vector(2, 0), 2, 3)  # Short focal length
    scene.add_optical_element(eyepiece)
    
    # Add distant object (parallel rays)
    scene.add_light_source(LightSource(Vector(-20, 1), num_rays=7, angle_range=(-0.01, 0.01)))
    scene.add_light_source(LightSource(Vector(-20, -1), num_rays=7, angle_range=(-0.01, 0.01)))
    
    return scene

def create_eye_system(simulator):
    """Create a scene simulating the human eye."""
    scene = simulator.create_scene()
    
    class Eye(OpticalElement):
        """A simplified model of the human eye."""
        
        def __init__(self, position, size=2.4, lens_power=20):
            """
            Initialize an eye model.
            
            Args:
                position (Vector): Position of the eye center
                size (float): Size/diameter of the eye
                lens_power (float): Power of the lens (inverse of focal length)
            """
            super().__init__(position)
            self.size = size
            self.lens_power = lens_power
            self.radius = size / 2
            
            # Cornea position (front of the eye)
            self.cornea_pos = Vector(position.x - 0.9*self.radius, position.y)
            
            # Lens position (slightly behind cornea)
            self.lens_pos = Vector(position.x - 0.7*self.radius, position.y)
            self.lens_width = 0.4 * size
            
            # Retina position (back of the eye)
            self.retina_pos = Vector(position.x + 0.9*self.radius, position.y)
        
        def interact(self, ray):
            """Handle ray interaction with the eye model."""
            # Check if ray intersects with the eye sphere
            oc = ray.position - self.position
            a = ray.direction.dot(ray.direction)
            b = 2.0 * oc.dot(ray.direction)
            c = oc.dot(oc) - self.radius**2
            
            discriminant = b**2 - 4*a*c
            
            if discriminant < 0:
                return  # No intersection
            
            # Find entrance point (cornea)
            t_cornea = (-b - math.sqrt(discriminant)) / (2.0 * a)
            
            if t_cornea <= 0:
                return  # Behind the ray
            
            # Move ray to cornea
            cornea_point = ray.position + ray.direction * t_cornea
            ray.position = cornea_point
            ray.path.append((ray.position.x, ray.position.y))
            
            # Simplified refraction at cornea (just use lens effect)
            # For an embedded lens, just use ray transfer through the lens position
            
            # Calculate distance from lens position to ray line
            lens_to_ray = ray.position - self.lens_pos
            lens_distance = lens_to_ray.dot(Vector(0, 1))  # Vertical distance
            
            # Move ray to lens position horizontally
            dx = self.lens_pos.x - ray.position.x
            ray.position = ray.position + ray.direction * (dx / ray.direction.x if ray.direction.x != 0 else 0)
            ray.path.append((ray.position.x, ray.position.y))
            
            # Apply lens refraction (simplified model)
            # Using thin lens equation: angle_deviation = -h/f where h is height from optical axis
            angle_deviation = -lens_distance / (1.0 / self.lens_power)
            
            # Rotate ray direction
            cos_a = math.cos(angle_deviation)
            sin_a = math.sin(angle_deviation)
            
            x = ray.direction.x
            y = ray.direction.y
            ray.direction = Vector(x * cos_a - y * sin_a, x * sin_a + y * cos_a).normalize()
            
            # Continue ray to retina or until it exits the eye
            # Find exit point (if any)
            oc = ray.position - self.position
            a = ray.direction.dot(ray.direction)
            b = 2.0 * oc.dot(ray.direction)
            c = oc.dot(oc) - self.radius**2
            
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0:
                t_exit = (-b + math.sqrt(discriminant)) / (2.0 * a)
                
                if t_exit > 0:
                    # Check if ray hits retina first
                    t_retina = (self.retina_pos.x - ray.position.x) / ray.direction.x if ray.direction.x != 0 else float('inf')
                    
                    if 0 < t_retina < t_exit:
                        # Ray hits retina
                        retina_point = ray.position + ray.direction * t_retina
                        ray.position = retina_point
                        ray.path.append((ray.position.x, ray.position.y))
                        ray.active = False  # Ray stops at retina
                    else:
                        # Ray exits eye
                        exit_point = ray.position + ray.direction * t_exit
                        ray.position = exit_point
                        ray.path.append((ray.position.x, ray.position.y))
        
        def draw(self, ax):
            """Draw the eye model on the plot."""
            # Draw eye outline
            circle = plt.Circle(
                (self.position.x, self.position.y), 
                self.radius, 
                fill=False, 
                edgecolor='black'
            )
            ax.add_patch(circle)
            
            # Draw cornea
            cornea_arc = plt.matplotlib.patches.Arc(
                (self.cornea_pos.x, self.cornea_pos.y),
                width=0.8*self.size, 
                height=self.size,
                theta1=-90, 
                theta2=90,
                edgecolor='blue',
                linestyle='-'
            )
            ax.add_patch(cornea_arc)
            
            # Draw lens
            lens_line = plt.Line2D(
                [self.lens_pos.x, self.lens_pos.x],
                [self.lens_pos.y - self.lens_width/2, self.lens_pos.y + self.lens_width/2],
                color='purple',
                linewidth=2
            )
            ax.add_line(lens_line)
            
            # Draw retina
            retina_arc = plt.matplotlib.patches.Arc(
                (self.retina_pos.x, self.retina_pos.y),
                width=0.8*self.size, 
                height=self.size,
                theta1=90, 
                theta2=270,
                edgecolor='red',
                linestyle='-'
            )
            ax.add_patch(retina_arc)
    
    # Add an eye to the scene
    eye = Eye(Vector(2, 0), size=4, lens_power=15)
    scene.add_optical_element(eye)
    
    # Add objects at different distances
    scene.add_light_source(LightSource(Vector(-6, 1), num_rays=3, angle_range=(-math.pi/10, math.pi/10)))
    scene.add_light_source(LightSource(Vector(-6, -1), num_rays=3, angle_range=(-math.pi/10, math.pi/10)))
    
    return scene