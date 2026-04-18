"""
QUADRUPED CO-EVOLUTION V3 — NICHE SPECIATION + FEET-GATED FITNESS
==================================================================

KEY CHANGES OVER V2
-------------------

1. FEET-GATED FITNESS (the structural fix)
   Distance reward is now MULTIPLIED by a "feet usage" factor in [0, 1].
   If the creature never lifted a foot, feet_factor = 0 and distance gives
   nothing. Walking is no longer "bonus on top of sliding" — it's the gate
   through which any score flows.

   feet_factor = min(1.0, touchdowns / TARGET_TOUCHDOWNS)

   A creature needs ~15 distinct foot touchdowns in a generation to get
   full credit for distance. Fewer touchdowns means proportionally less
   credit. Zero touchdowns means zero fitness regardless of distance.

2. NICHE SPECIATION
   Instead of clustering brains by weight centroid, species are now defined
   by which fitness-component a creature excelled at. Four niches:
     - DISTANCE specialists  (best at raw x-movement)
     - FOOTWORK specialists  (best at alternating touchdowns)
     - STABILITY specialists (best upright frames / least drag)
     - EFFICIENCY specialists (best fitness-per-energy ratio)
   Every generation, the top creature in each niche is automatically
   preserved as an elite, even if their overall fitness isn't the highest.
   This prevents single-strategy convergence.

3. LIFTED-FEET BONUS (reward time in the air, not just contact)
   An airborne foot is rewarded slightly — teaches the brain that feet
   are *meant* to move, not just drag on the ground. Prevents the
   stationary-feet-on-ground exploit (earn foot_contact without walking).

4. CRAWLING/SLIDING DETECTION
   A new "slide frames" counter tracks frames where the torso is moving
   forward BUT no feet just touched down. High slide_frames is penalised.
   This catches the "hover and drag" strategy the population found.

5. SURVIVORSHIP BIAS FIX
   Better speciation tracking and a clearer "niche HUD" showing which
   creature leads each niche.
"""

import pygame
import pymunk
import pymunk.pygame_util
import random
import math
import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

pygame.init()
pygame.event.set_allowed([pygame.QUIT, pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN])
WIDTH, HEIGHT = 2400, 1080
FPS = 60
GEN_TIME = 25
POP_SIZE = 30
FF_MULTIPLIER = 8
SUBSTEPS = 6
MAX_LIN_VEL = 800
MAX_ANG_VEL = 10
CURRENT_GEN = 1             # updated each generation in main()

CAT_GROUND = 0x1
CAT_WALKER = 0x2

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
BONE_COLOR   = (240, 235, 220)
DEAD_COLOR   = (110, 60, 60)
BTN_BG       = (40, 40, 60)
BTN_HI       = (90, 50, 50)
SKY_COLOR    = (14, 14, 22)
GROUND_COLOR = (45, 48, 55)
JOINT_DOT    = (50, 220, 255)
JOINT_LEADER = (255, 255, 120)
HEAD_COLOR   = (200, 160, 255)
TAIL_COLOR   = (255, 150, 200)
SPINE_DOT    = (255, 130, 80)
FOOT_DOT     = (80, 255, 120)
AVG_COLOR    = (0, 150, 255)
WORST_COLOR  = (255, 100, 0)
WINNER_COLOR = (255, 255, 0)

# --- NICHE SPECIATION ---
NICHES = ["distance", "footwork", "stability", "efficiency"]
NICHE_COLORS = {
    "distance":   (80, 255, 120),    # green
    "footwork":   (80, 180, 255),    # blue
    "stability":  (255, 220, 80),    # yellow
    "efficiency": (255, 120, 220),   # pink
}
NICHE_SLOTS = {
    "distance":   6,  # most slots for raw progress
    "footwork":   12,  # equally important: gait quality
    "stability":  9,
    "efficiency": 3,
}
# Total must equal POP_SIZE
assert sum(NICHE_SLOTS.values()) == POP_SIZE

NICHE_ABBREV = {"distance": "DIST", "footwork": "FOOT", "stability": "STAB", "efficiency": "EFF"}

STATE_SIM      = 0
STATE_STATS    = 1
STATE_BRAIN    = 2
STATE_GAIT     = 3
STATE_FITNESS  = 4
STATE_BODY     = 5

# --- BODY PARAMS ---
BODY_PARAM_BOUNDS = {
    "torso_rear_len":     (0.7, 1.3), "torso_rear_h":       (0.8, 1.2),
    "torso_rear_mass":    (0.6, 1.5), "torso_front_len":    (0.7, 1.3),
    "torso_front_h":      (0.8, 1.2), "torso_front_mass":   (0.6, 1.5),
    "spine_limit":        (0.5, 1.6), "spine_stiffness":    (0.4, 2.0),
    "spine_damping":      (0.4, 2.0), "spine_motor_force":  (0.5, 1.8),
    "head_radius":        (0.8, 1.2), "head_mass":          (0.7, 1.4),
    "neck_limit":         (0.7, 1.4),
    "tail_seg_len":       (0.7, 1.3), "tail_seg_mass":      (0.6, 1.5),
    "tail_stiffness":     (0.4, 2.0),
    "upper_leg_len":      (0.7, 1.3), "upper_leg_mass":     (0.6, 1.5),
    "lower_leg_len":      (0.7, 1.3), "lower_leg_mass":     (0.6, 1.5),
    "foot_len":           (0.7, 1.3), "foot_mass":          (0.6, 1.5),
    "foot_friction":      (0.7, 1.3),
    "hip_limit_fwd":      (0.7, 1.4), "hip_limit_back":     (0.7, 1.4),
    "knee_limit":         (0.7, 1.4),
}

DEFAULT_BODY = {
    "torso_rear_len": 140, "torso_rear_h": 50, "torso_rear_mass": 30,
    "torso_front_len": 120, "torso_front_h": 40, "torso_front_mass": 18,
    "spine_limit": 0.6, "spine_stiffness": 8000, "spine_damping": 400,
    "spine_motor_force": 3e6,
    "head_radius": 22, "head_mass": 6, "neck_limit": 0.5,
    "tail_segments": 3, "tail_seg_len": 45, "tail_seg_mass": 1.0,
    "tail_stiffness": 600,
    "upper_leg_len": 70, "upper_leg_mass": 4.5,
    "lower_leg_len": 100, "lower_leg_mass": 2.5,
    "foot_len": 50, "foot_mass": 1.2, "foot_friction": 4.0,
    "hip_limit_fwd": 0.9, "hip_limit_back": 0.6, "knee_limit": 1.6,
    "gravity": 2000,
}

BODY_MUTATION_RATE  = 0.15
BODY_MUTATION_SIGMA = 0.06

# ---------------------------------------------------------------------------
# FITNESS CONFIG — RESTRUCTURED
# ---------------------------------------------------------------------------
# Feet are now a GATE: distance is multiplied by feet_factor ∈ [0, 1].
# At TARGET_TOUCHDOWNS (15) feet_factor saturates at 1.0.
# No touchdowns = no distance credit, period.

TARGET_TOUCHDOWNS        = 15
FIT_DISTANCE_GAIN        = 1.0
FIT_ALTERNATING_GAIN     = 0.80   # 0.80 per touchdown (direct reward, not gated)
FIT_LIFTED_FOOT_GAIN     = 0.04 # per frame per airborne foot (encourages leg motion)
FIT_UPRIGHT_GAIN         = 0.04   # slight upright bonus (reduced, no longer the exploit)
FIT_SLIDE_PENALTY        = 0.15   # per frame of sliding (moving forward without foot touchdowns)
FIT_TORSO_DRAG_PENALTY   = 0.40   # stronger than before
FIT_HEAD_DRAG_PENALTY    = 0.60
FIT_TILT_PENALTY         = 0.25
FIT_ENERGY_PENALTY       = 0.003


# ---------------------------------------------------------------------------
# RNN BRAIN
# ---------------------------------------------------------------------------

class RNNBrain:
    def __init__(self, weights=None):
        self.input_size, self.hidden_size, self.output_size = 22, 16, 10
        if weights is not None:
            self.w_in, self.w_rec, self.w_out = weights
        else:
            self.w_in  = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
            self.w_rec = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.hidden_size))
            self.w_out = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.h_state = np.zeros(self.hidden_size)

    def predict(self, inputs):
        combined = np.dot(inputs, self.w_in) + np.dot(self.h_state, self.w_rec)
        self.h_state = np.tanh(combined)
        return np.tanh(np.dot(self.h_state, self.w_out))

    def mutate(self, rate=0.05):
        new_weights = []
        for w in [self.w_in, self.w_rec, self.w_out]:
            mask = np.random.rand(*w.shape) < rate
            noise = np.random.normal(0, 0.2, w.shape)
            new_weights.append(w + mask * noise)
        w_out = new_weights[2]
        for l_idx, r_idx in [(2, 4), (3, 5), (6, 8), (7, 9)]:
            avg = (w_out[:, l_idx] + w_out[:, r_idx]) / 2.0
            w_out[:, l_idx] = w_out[:, l_idx] * 0.8 + avg * 0.2
            w_out[:, r_idx] = w_out[:, r_idx] * 0.8 + avg * 0.2
        return RNNBrain(new_weights)

    def save(self, path):
        np.savez(path, w_in=self.w_in, w_rec=self.w_rec, w_out=self.w_out)

    @staticmethod
    def load(path):
        data = np.load(path)
        return RNNBrain([data["w_in"], data["w_rec"], data["w_out"]])


# ---------------------------------------------------------------------------
# GENOME
# ---------------------------------------------------------------------------

class Genome:
    def __init__(self, brain, body_params, midpoint):
        self.brain = brain
        self.body = dict(body_params)
        self._midpoint = midpoint

    def mutate_body(self):
        new_body = dict(self.body)
        for key, (fmin, fmax) in BODY_PARAM_BOUNDS.items():
            if random.random() < BODY_MUTATION_RATE:
                mid = self._midpoint[key]
                low, high = mid * fmin, mid * fmax
                noise = random.gauss(0, BODY_MUTATION_SIGMA) * mid
                new_val = new_body[key] + noise
                new_val = max(low, min(high, new_val))
                new_body[key] = new_val
        for key in self.body:
            if key not in BODY_PARAM_BOUNDS:
                new_body[key] = self.body[key]
        return new_body

    def clone(self, mutate_brain_rate=0.05):
        return Genome(self.brain.mutate(mutate_brain_rate), self.mutate_body(), self._midpoint)


def genome_crossover(a, b):
    new_weights = []
    for w1, w2 in zip([a.brain.w_in, a.brain.w_rec, a.brain.w_out],
                      [b.brain.w_in, b.brain.w_rec, b.brain.w_out]):
        mask = np.random.rand(*w1.shape) > 0.5
        new_weights.append(np.where(mask, w1, w2))
    new_brain = RNNBrain(new_weights)
    new_body = {}
    for key in a.body:
        if key in BODY_PARAM_BOUNDS:
            new_body[key] = (a.body[key] + b.body[key]) / 2 if random.random() < 0.3 else random.choice([a.body[key], b.body[key]])
        else:
            new_body[key] = a.body[key]
    return Genome(new_brain, new_body, a._midpoint)


# ---------------------------------------------------------------------------
# NICHE-BASED EVOLUTION
# ---------------------------------------------------------------------------

def niche_score(q, niche):
    """How well does this creature score in a particular niche?"""
    if niche == "distance":
        return max(0.0, q.bodies[0].position.x - 600)
    elif niche == "footwork":
        # Raw touchdown count + alternation bonus (captured in touchdown_events)
        return q.foot_touchdown_events
    elif niche == "stability":
        # Upright frames minus drag frames
        return q.upright_frames - q.torso_drag_frames - q.head_drag_frames
    elif niche == "efficiency":
        # Fitness per unit energy; avoid div-by-zero
        energy = max(1.0, q.total_energy_used)
        return q.cached_fitness / energy
    return 0.0


def assign_niches(population):
    """Return dict niche -> list of creatures sorted by that niche's score desc."""
    niche_rankings = {}
    for niche in NICHES:
        sorted_pop = sorted(population, key=lambda q: niche_score(q, niche), reverse=True)
        niche_rankings[niche] = sorted_pop
    return niche_rankings


def live_niche_score(q, niche):
    """Current-state score used only for LEADERS rendering.
    Uses instantaneous physical state, not accumulated history."""
    if niche == "distance":
        return q.bodies[0].position.x
    elif niche == "footwork":
        return q.foot_touchdown_events
    elif niche == "stability":
        return -(q.bodies[0].position.y)  # lowest Y = highest off ground = most upright
    elif niche == "efficiency":
        return q.cached_fitness / max(1.0, q.total_energy_used)
    return 0.0


def assign_live_leaders(alive_pop):
    """Returns dict niche -> single leader creature by current physical state.
    Used only for LEADERS display — does not affect evolution."""
    if not alive_pop:
        return {}
    return {niche: max(alive_pop, key=lambda q: live_niche_score(q, niche))
            for niche in NICHES}


def build_next_generation_niche(population):
    """
    Build the next generation with niche-based speciation.
    Each niche contributes NICHE_SLOTS[niche] individuals.
    Top of each niche survives as elite. Rest of niche slots filled via
    intra-niche crossover and mutation.
    """
    niche_rankings = assign_niches(population)
    new_genomes = []

    for niche, slots in NICHE_SLOTS.items():
        ranked = niche_rankings[niche]
        # Only consider creatures with non-zero niche score for breeding
        viable = [q for q in ranked if niche_score(q, niche) > 0]
        if not viable:
            # Fallback: use top creatures by overall fitness
            viable = sorted(population, key=lambda q: q.cached_fitness, reverse=True)[:5]

        # Elite: top creature in this niche
        elite = viable[0]
        elite.niche_tag = niche
        new_genomes.append(elite.genome)
        remaining = slots - 1

        # Take top 5 from this niche as breeding pool
        pool = [q.genome for q in viable[:min(5, len(viable))]]
        for _ in range(remaining):
            if len(pool) >= 2 and random.random() < 0.5:
                p1, p2 = random.sample(pool, 2)
                child = genome_crossover(p1, p2).clone(mutate_brain_rate=0.04)
            else:
                child = random.choice(pool).clone(mutate_brain_rate=0.08)
            new_genomes.append(child)

    # Safety: pad if something went wrong
    while len(new_genomes) < POP_SIZE:
        new_genomes.append(random.choice(population).genome.clone(mutate_brain_rate=0.1))

    return new_genomes[:POP_SIZE]


def tag_creatures_by_niche(population):
    """Assign each creature its 'home' niche: the one where it ranks best.
    Used for colouring in the sim view."""
    rankings = assign_niches(population)
    for q in population:
        best_niche, best_rank = None, 1e9
        for niche in NICHES:
            rank = rankings[niche].index(q) if q in rankings[niche] else 1e9
            if rank < best_rank:
                best_rank = rank
                best_niche = niche
        q.niche_tag = best_niche or "distance"


# ---------------------------------------------------------------------------
# CURRICULUM
# ---------------------------------------------------------------------------

def curriculum_weights(gen):
    """Returns per-component fitness multipliers keyed by stage."""
    if gen <= 50:       # STAND
        return {"distance": 0.0, "alt": 0.0, "lifted": 0.0, "upright": 5.0, "penalties": 1.0}
    elif gen <= 150:    # BALANCE
        return {"distance": 0.0, "alt": 0.5, "lifted": 1.0, "upright": 2.0, "penalties": 1.0}
    else:               # WALK
        return {"distance": 1.0, "alt": 1.0, "lifted": 1.0, "upright": 1.0, "penalties": 1.0}


def curriculum_stage(gen):
    """Returns (stage_name, colour) for HUD display."""
    if gen <= 50:
        return "STAND",   (255, 140,   0)   # orange
    elif gen <= 150:
        return "BALANCE", (255, 255,   0)   # yellow
    else:
        return "WALK",    (  0, 255,   0)   # green


# ---------------------------------------------------------------------------
# QUADRUPED
# ---------------------------------------------------------------------------

class Quadruped:
    def __init__(self, space, genome):
        self.space = space
        self.genome = genome
        self.body_params = genome.body
        self.nn = genome.brain

        self.niche_tag = None
        self.bodies = []
        self.shapes = []
        self.constraints = []
        self.motors = []
        self.feet_shapes = []
        self.torso_shapes = []
        self.upper_leg_shapes = []
        self.joint_positions = []
        self.is_dead = False
        self.is_glitched = False

        self.total_tilt = 0.0
        self.total_energy_used = 0.0
        self.foot_contact_frames = 0
        self.foot_touchdown_events = 0
        self.lifted_foot_frames = 0      # NEW: sum of airborne-foot-frames across all feet
        self.upright_frames = 0
        self.torso_drag_frames = 0
        self.head_drag_frames = 0
        self.slide_frames = 0            # NEW: frames where moving forward but no feet engaged
        self.unique_foot_sequence = []

        self._prev_foot_contact = [False, False, False, False]
        self._last_touchdown_frame = -1
        self._frame_count = 0
        self.last_x = 600
        self.cached_fitness = 0.0
        self._input_buf = np.zeros(22)
        self.last_motor_out = np.zeros(10)

        self.build_body()

    def _p(self, key):
        return self.body_params[key]

    def build_body(self):
        sx, sy = 600, HEIGHT - 450
        wf = pymunk.ShapeFilter(group=random.randint(1, 99999),
                                 categories=CAT_WALKER, mask=CAT_GROUND)
        p = self._p

        rear_len, rear_h, rear_m = p("torso_rear_len"), p("torso_rear_h"), p("torso_rear_mass")
        front_len, front_h, front_m = p("torso_front_len"), p("torso_front_h"), p("torso_front_mass")

        m_rear = pymunk.Body(rear_m, pymunk.moment_for_box(rear_m, (rear_len, rear_h)))
        m_rear.position = (sx, sy)
        m_rear.sleep_time_threshold = float('inf')
        s_rear = pymunk.Poly.create_box(m_rear, (rear_len, rear_h))
        s_rear._is_circle = False
        s_rear.filter = wf
        s_rear.friction = 0.1
        self.torso_shapes.append(s_rear)

        m_front = pymunk.Body(front_m, pymunk.moment_for_box(front_m, (front_len, front_h)))
        m_front.position = (sx + rear_len / 2 + front_len / 2, sy)
        m_front.sleep_time_threshold = float('inf')
        s_front = pymunk.Poly.create_box(m_front, (front_len, front_h))
        s_front._is_circle = False
        s_front.filter = wf
        s_front.friction = 0.1
        self.torso_shapes.append(s_front)

        spine_anchor = (sx + rear_len / 2, sy)
        sp_p = pymunk.PivotJoint(m_rear, m_front, spine_anchor)
        sp_l = pymunk.RotaryLimitJoint(m_rear, m_front, -p("spine_limit"), p("spine_limit"))
        sp_s = pymunk.DampedRotarySpring(m_rear, m_front, rest_angle=0.0,
                                          stiffness=p("spine_stiffness"),
                                          damping=p("spine_damping"))
        sp_m = pymunk.SimpleMotor(m_rear, m_front, 0)
        sp_m.max_force = min(p("spine_motor_force"), 2e5)
        self.space.add(m_rear, s_rear, m_front, s_front, sp_p, sp_l, sp_s, sp_m)
        self.bodies.extend([m_rear, m_front])
        self.shapes.extend([s_rear, s_front])
        self.constraints.extend([sp_p, sp_l, sp_s, sp_m])
        self.motors.append(sp_m)
        _rl = rear_len / 2
        self.joint_positions.append(("spine",
            lambda rb=m_rear, x=_rl: rb.position + pymunk.Vec2d(x, 0).rotated(rb.angle)))

        # Head
        head_r = p("head_radius")
        head_pos = (m_front.position[0] + front_len / 2 + head_r + 10,
                    sy - front_h / 2 - head_r)
        self.head_body = pymunk.Body(p("head_mass"),
                                     pymunk.moment_for_circle(p("head_mass"), 0, head_r))
        self.head_body.position = head_pos
        self.head_body.sleep_time_threshold = float('inf')
        self.head_shape = pymunk.Circle(self.head_body, head_r)
        self.head_shape._is_circle = True
        self.head_shape.filter = wf
        self.head_shape.friction = 0.1
        neck_p = pymunk.PivotJoint(m_front, self.head_body,
                                    (front_len / 2, -front_h / 2),
                                    (-head_r * 0.8, head_r * 0.8))
        neck_l = pymunk.RotaryLimitJoint(m_front, self.head_body,
                                          -p("neck_limit"), p("neck_limit"))
        neck_s = pymunk.DampedRotarySpring(m_front, self.head_body, rest_angle=0.0,
                                            stiffness=2000, damping=200)
        neck_m = pymunk.SimpleMotor(m_front, self.head_body, 0)
        neck_m.max_force = 1e5
        self.space.add(self.head_body, self.head_shape, neck_p, neck_l, neck_s, neck_m)
        self.bodies.append(self.head_body)
        self.shapes.append(self.head_shape)
        self.constraints.extend([neck_p, neck_l, neck_s, neck_m])
        self.motors.append(neck_m)
        _nl = (front_len / 2, -front_h / 2)
        self.joint_positions.append(("neck",
            lambda fb=m_front, a=_nl: fb.position + pymunk.Vec2d(a[0], a[1]).rotated(fb.angle)))

        # Tail
        n_tail = int(p("tail_segments"))
        tail_len, tail_mass, tail_stiff = p("tail_seg_len"), p("tail_seg_mass"), p("tail_stiffness")
        prev_body = m_rear
        attach_local = (-rear_len / 2, -rear_h / 4)
        for i in range(n_tail):
            seg_w = tail_len
            seg_h = max(6, 14 - i * 2)
            seg_pos = (sx - rear_len / 2 - seg_w / 2 - i * seg_w, sy - rear_h / 4)
            seg_body = pymunk.Body(tail_mass, pymunk.moment_for_box(tail_mass, (seg_w, seg_h)))
            seg_body.position = seg_pos
            seg_body.sleep_time_threshold = float('inf')
            seg_shape = pymunk.Poly.create_box(seg_body, (seg_w, seg_h))
            seg_shape._is_circle = False
            seg_shape.filter = wf
            seg_shape.friction = 0.1
            anchor_prev = attach_local if i == 0 else (-seg_w / 2, 0)
            anchor_cur = (seg_w / 2, 0)
            j = pymunk.PivotJoint(prev_body, seg_body, anchor_prev, anchor_cur)
            jl = pymunk.RotaryLimitJoint(prev_body, seg_body, -0.5, 0.5)
            js = pymunk.DampedRotarySpring(prev_body, seg_body, rest_angle=0.0,
                                            stiffness=tail_stiff, damping=tail_stiff * 0.08)
            self.space.add(seg_body, seg_shape, j, jl, js)
            self.bodies.append(seg_body)
            self.shapes.append(seg_shape)
            self.constraints.extend([j, jl, js])
            _pb, _ap = prev_body, anchor_prev
            self.joint_positions.append(("tail",
                lambda pb=_pb, a=_ap: pb.position + pymunk.Vec2d(a[0], a[1]).rotated(pb.angle)))
            prev_body = seg_body

        # Legs
        upper_len, upper_m = p("upper_leg_len"), p("upper_leg_mass")
        lower_len, lower_m = p("lower_leg_len"), p("lower_leg_mass")
        foot_len, foot_m = p("foot_len"), p("foot_mass")
        foot_fric = p("foot_friction")
        hip_fwd, hip_back = p("hip_limit_fwd"), p("hip_limit_back")
        knee_lim = p("knee_limit")

        leg_specs = [
            ("RL", m_rear,  -rear_len * 0.35, rear_h / 2),
            ("RR", m_rear,   rear_len * 0.35, rear_h / 2),
            ("FL", m_front, -front_len * 0.35, front_h / 2),
            ("FR", m_front,  front_len * 0.35, front_h / 2),
        ]

        for idx, (name, parent, lx, ly) in enumerate(leg_specs):
            hip_world = parent.position + (lx, ly)
            upper_pos = (hip_world[0], hip_world[1] + upper_len / 2)
            u_body = pymunk.Body(upper_m, pymunk.moment_for_box(upper_m, (14, upper_len)))
            u_body.position = upper_pos
            u_body.sleep_time_threshold = float('inf')
            u_shape = pymunk.Poly.create_box(u_body, (14, upper_len))
            u_shape._is_circle = False
            u_shape.filter = wf
            u_shape.friction = 0.1
            self.upper_leg_shapes.append(u_shape)

            knee_world = (hip_world[0], hip_world[1] + upper_len)
            lower_pos = (knee_world[0], knee_world[1] + lower_len / 2)
            l_body = pymunk.Body(lower_m, pymunk.moment_for_box(lower_m, (10, lower_len)))
            l_body.position = lower_pos
            l_body.sleep_time_threshold = float('inf')
            l_shape = pymunk.Poly.create_box(l_body, (10, lower_len))
            l_shape._is_circle = False
            l_shape.filter = wf
            l_shape.friction = 0.1

            ankle_world = (knee_world[0], knee_world[1] + lower_len)
            foot_pos = (ankle_world[0] + foot_len / 3, ankle_world[1])
            f_body = pymunk.Body(foot_m, pymunk.moment_for_box(foot_m, (foot_len, 10)))
            f_body.position = foot_pos
            f_body.sleep_time_threshold = float('inf')
            f_shape = pymunk.Poly.create_box(f_body, (foot_len, 10))
            f_shape._is_circle = False
            f_shape.filter = wf
            f_shape.friction = foot_fric

            hip_p = pymunk.PivotJoint(parent, u_body, hip_world)
            hip_l = pymunk.RotaryLimitJoint(parent, u_body, -hip_back, hip_fwd)
            hip_m = pymunk.SimpleMotor(parent, u_body, 0)
            hip_m.max_force = 3e5

            knee_p = pymunk.PivotJoint(u_body, l_body, knee_world)
            knee_l = pymunk.RotaryLimitJoint(u_body, l_body, -0.1, knee_lim)
            knee_m = pymunk.SimpleMotor(u_body, l_body, 0)
            knee_m.max_force = 2e5

            ankle_p = pymunk.PivotJoint(l_body, f_body, ankle_world)
            ankle_l = pymunk.RotaryLimitJoint(l_body, f_body, -0.4, 0.4)
            ankle_s = pymunk.DampedRotarySpring(l_body, f_body, rest_angle=0.0,
                                                 stiffness=3000, damping=300)

            self.space.add(u_body, u_shape, l_body, l_shape, f_body, f_shape,
                           hip_p, hip_l, hip_m, knee_p, knee_l, knee_m,
                           ankle_p, ankle_l, ankle_s)
            self.bodies.extend([u_body, l_body, f_body])
            self.shapes.extend([u_shape, l_shape, f_shape])
            self.constraints.extend([hip_p, hip_l, hip_m, knee_p, knee_l, knee_m,
                                      ankle_p, ankle_l, ankle_s])
            self.motors.extend([hip_m, knee_m])
            self.feet_shapes.append(f_shape)

            _parent = parent
            _hl = (lx, ly)
            self.joint_positions.append(("hip",
                lambda pb=_parent, a=_hl: pb.position + pymunk.Vec2d(a[0], a[1]).rotated(pb.angle)))
            _ub, _ul = u_body, upper_len
            self.joint_positions.append(("knee",
                lambda pb=_ub, u=_ul: pb.position + pymunk.Vec2d(0, u / 2).rotated(pb.angle)))
            _lb, _ll = l_body, lower_len
            self.joint_positions.append(("ankle",
                lambda pb=_lb, u=_ll: pb.position + pymunk.Vec2d(0, u / 2).rotated(pb.angle)))

    def _kill(self):
        self.is_dead = True
        for m in self.motors:
            m.rate = 0.0
        self.nn.h_state = np.zeros(self.nn.hidden_size)

    def _check_ground_contact(self, shape, ground_y, margin=5):
        return shape.bb.bottom >= ground_y - margin

    def update(self, t, ground_y):
        if self.is_dead:
            return
        self._frame_count += 1
        torso = self.bodies[0]
        self.total_tilt += abs(torso.angle)

        buf = self._input_buf
        buf[0] = torso.angle
        buf[1] = torso.angular_velocity
        buf[2] = self.head_body.angle
        buf[3] = self.head_body.position.y / HEIGHT
        buf[4] = math.sin(t)
        n_motors = min(len(self.motors), 10)
        for idx in range(n_motors):
            buf[5 + idx] = self.motors[idx].rate / 10.0
        for idx in range(10 - n_motors):
            buf[5 + n_motors + idx] = 0.0
        foot_contacts_now = []
        for idx, f in enumerate(self.feet_shapes[:4]):
            contact = self._check_ground_contact(f, ground_y, margin=8)
            foot_contacts_now.append(contact)
            buf[5 + 10 + idx] = 1.0 if contact else 0.0
        for idx in range(4 - min(4, len(self.feet_shapes))):
            buf[5 + 10 + len(self.feet_shapes) + idx] = 0.0
        for idx in range(22 - 19):
            buf[19 + idx] = 0.0

        outputs = self.nn.predict(buf)
        self.last_motor_out = outputs

        # Soft-start: first 30 frames let the body settle; brain still runs
        if self._frame_count >= 30:
            for i in range(min(len(self.motors), 10)):
                self.motors[i].rate = float(np.clip(outputs[i], -1.0, 1.0)) * 3.0
                self.total_energy_used += abs(self.motors[i].rate) * FIT_ENERGY_PENALTY
        else:
            for i in range(min(len(self.motors), 10)):
                self.motors[i].rate = 0.0

        # Velocity clamping — preserves direction, only reduces magnitude
        for body in self.bodies:
            if body.velocity.length > MAX_LIN_VEL:
                body.velocity = body.velocity.normalized() * MAX_LIN_VEL
            if abs(body.angular_velocity) > MAX_ANG_VEL:
                body.angular_velocity = math.copysign(MAX_ANG_VEL, body.angular_velocity)

        # --- FITNESS TRACKING ---
        any_foot_down = any(foot_contacts_now)
        if any_foot_down:
            self.foot_contact_frames += 1

        # Alternation and touchdown events
        any_new_touchdown = False
        for i, (now, was) in enumerate(zip(foot_contacts_now, self._prev_foot_contact)):
            if now and not was:
                self.foot_touchdown_events += 1
                any_new_touchdown = True
                if not self.unique_foot_sequence or self.unique_foot_sequence[-1] != i:
                    self.foot_touchdown_events += 1
                self.unique_foot_sequence.append(i)
                if len(self.unique_foot_sequence) > 20:
                    self.unique_foot_sequence.pop(0)
                self._last_touchdown_frame = self._frame_count
        self._prev_foot_contact = foot_contacts_now

        # LIFTED FOOT REWARD — encourage airborne feet (= leg motion)
        airborne_count = sum(1 for c in foot_contacts_now if not c)
        self.lifted_foot_frames += airborne_count

        # Upright reward
        torso_height_above = ground_y - torso.position.y
        min_upright_height = self._p("upper_leg_len") * 0.4
        if torso_height_above > min_upright_height:
            self.upright_frames += 1

        # Drag penalties
        torso_on_ground = False
        for s in self.torso_shapes:
            if self._check_ground_contact(s, ground_y):
                torso_on_ground = True
                break
        if not torso_on_ground:
            for s in self.upper_leg_shapes:
                if self._check_ground_contact(s, ground_y):
                    torso_on_ground = True
                    break
        if torso_on_ground:
            self.torso_drag_frames += 1
        if self._check_ground_contact(self.head_shape, ground_y):
            self.head_drag_frames += 1

        # SLIDE DETECTION: moving forward but no recent foot touchdown
        forward_motion = (torso.position.x - self.last_x) > 0.5
        time_since_touchdown = self._frame_count - self._last_touchdown_frame
        if forward_motion and time_since_touchdown > 30:
            self.slide_frames += 1

        # --- COMPOSITE FITNESS (with feet gate + curriculum weights) ---
        cw = curriculum_weights(CURRENT_GEN)
        dist = max(0.0, torso.position.x - 600)
        feet_factor = min(1.0, self.foot_touchdown_events / TARGET_TOUCHDOWNS)
        distance_score = dist * FIT_DISTANCE_GAIN * feet_factor          * cw["distance"]
        alt_score      = self.foot_touchdown_events * FIT_ALTERNATING_GAIN * cw["alt"]
        lifted_score   = self.lifted_foot_frames * FIT_LIFTED_FOOT_GAIN    * cw["lifted"]
        upright_score  = self.upright_frames * FIT_UPRIGHT_GAIN            * cw["upright"]
        torso_pen      = self.torso_drag_frames * FIT_TORSO_DRAG_PENALTY   * cw["penalties"]
        head_pen       = self.head_drag_frames * FIT_HEAD_DRAG_PENALTY     * cw["penalties"]
        tilt_pen       = self.total_tilt * FIT_TILT_PENALTY                * cw["penalties"]
        slide_pen      = self.slide_frames * FIT_SLIDE_PENALTY             * cw["penalties"]

        self.cached_fitness = max(0.0,
            distance_score + alt_score + lifted_score + upright_score
            - torso_pen - head_pen - tilt_pen - slide_pen - self.total_energy_used
        )

        if abs(torso.position.x - self.last_x) > 400:
            self.is_glitched = True
            self._kill()
        self.last_x = torso.position.x
        if self.head_drag_frames > 30:
            self._kill()

    def fitness_components(self):
        dist = max(0.0, self.bodies[0].position.x - 600)
        feet_factor = min(1.0, self.foot_touchdown_events / TARGET_TOUCHDOWNS)
        return {
            "distance":    dist * FIT_DISTANCE_GAIN * feet_factor,
            "alt":         self.foot_touchdown_events * FIT_ALTERNATING_GAIN,
            "lifted":      self.lifted_foot_frames * FIT_LIFTED_FOOT_GAIN,
            "upright":     self.upright_frames * FIT_UPRIGHT_GAIN,
            "torso_pen":   self.torso_drag_frames * FIT_TORSO_DRAG_PENALTY,
            "head_pen":    self.head_drag_frames * FIT_HEAD_DRAG_PENALTY,
            "tilt_pen":    self.total_tilt * FIT_TILT_PENALTY,
            "slide_pen":   self.slide_frames * FIT_SLIDE_PENALTY,
            "energy_pen":  self.total_energy_used,
            "feet_factor": feet_factor,
        }

    def draw(self, sc, cam_x, body_col, joint_col, leader=False):
        for b, s in zip(self.bodies, self.shapes):
            if s._is_circle:
                pos = (int(b.position.x - cam_x), int(b.position.y))
                pygame.draw.circle(sc, body_col, pos, int(s.radius))
                pygame.draw.circle(sc, HEAD_COLOR, pos, int(s.radius), 3)
            else:
                verts = [v.rotated(b.angle) + b.position - (cam_x, 0) for v in s.get_vertices()]
                pygame.draw.polygon(sc, body_col, verts)
                pygame.draw.polygon(sc, (0, 0, 0), verts, 2)

        if self.is_dead:
            return

        dot_radius = 8 if leader else 6
        for kind, fn in self.joint_positions:
            pos = fn()
            draw_pos = (int(pos.x - cam_x), int(pos.y))
            if kind == "spine":
                colour, r = SPINE_DOT, dot_radius + 2
            else:
                colour, r = joint_col, dot_radius
            pygame.draw.circle(sc, (0, 0, 0), draw_pos, r + 2)
            pygame.draw.circle(sc, colour, draw_pos, r)

        for i, f in enumerate(self.feet_shapes):
            pos = f.body.position
            draw_pos = (int(pos.x - cam_x), int(pos.y))
            colour = FOOT_DOT if self._prev_foot_contact[i] else (80, 120, 80)
            r = 10 if self._prev_foot_contact[i] else 7
            pygame.draw.circle(sc, (0, 0, 0), draw_pos, r + 2)
            pygame.draw.circle(sc, colour, draw_pos, r)

    def cleanup(self):
        for c in self.constraints:
            if c in self.space.constraints:
                self.space.remove(c)
        for s in self.shapes:
            if s in self.space.shapes:
                self.space.remove(s)
        for b in self.bodies:
            if b in self.space.bodies:
                self.space.remove(b)


# ---------------------------------------------------------------------------
# HISTORY
# ---------------------------------------------------------------------------

class History:
    def __init__(self):
        self.data = []
        self.fitness_components_log = []
        self.brain_log = []
        self.max_log = 300
        self.gait_trail = []
        self.body_log = []
        self.niche_leaders_log = []     # NEW: per-gen dict of niche -> leader fitness

    def record(self, population):
        alive = [p for p in population if not p.is_dead]
        best = max(p.cached_fitness for p in population)
        avg = sum(p.cached_fitness for p in population) / len(population)
        worst = min(p.cached_fitness for p in alive) if alive else 0
        dead = len(population) - len(alive)
        self.data.append((best, avg, worst, dead))

        best_walker = max(population, key=lambda p: p.cached_fitness)
        self.fitness_components_log.append(best_walker.fitness_components())
        self.body_log.append(dict(best_walker.body_params))

        # Niche leaders
        rankings = assign_niches(population)
        self.niche_leaders_log.append({
            niche: niche_score(rankings[niche][0], niche) for niche in NICHES
        })

    def log_brain(self, motor_outputs):
        self.brain_log.append(list(motor_outputs))
        if len(self.brain_log) > self.max_log:
            self.brain_log.pop(0)
        m = motor_outputs
        pca_x = (m[2] + m[3] + m[6] + m[7]) - (m[4] + m[5] + m[8] + m[9])
        pca_y = (m[2] + m[4] + m[6] + m[8]) - (m[3] + m[5] + m[7] + m[9])
        self.gait_trail.append([pca_x, pca_y, 255])
        if len(self.gait_trail) > 500:
            self.gait_trail.pop(0)
        for p in self.gait_trail:
            p[2] -= 2
            if p[2] < 0:
                p[2] = 0

    def draw_stats(self, screen, font):
        screen.fill((15, 15, 20))
        if len(self.data) < 2:
            msg = font.render("Not enough data.", True, WHITE)
            screen.blit(msg, (WIDTH // 2 - 200, HEIGHT // 2))
            return
        m_l, m_r, m_t, m_b = 200, 120, 160, 160
        g_w, g_h = WIDTH - m_l - m_r, HEIGHT - m_t - m_b
        title = font.render("FITNESS OVER GENERATIONS", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 35))
        pygame.draw.line(screen, WHITE, (m_l, HEIGHT - m_b), (WIDTH - m_r, HEIGHT - m_b), 3)
        pygame.draw.line(screen, WHITE, (m_l, m_t), (m_l, HEIGHT - m_b), 3)
        max_val = max([d[0] for d in self.data] + [100])
        dx = g_w / max(len(self.data) - 1, 1)
        for tick in range(0, int(max_val) + 1, max(1, int(max_val // 6))):
            y = (HEIGHT - m_b) - (tick / max_val * g_h)
            pygame.draw.line(screen, (60, 60, 60), (m_l, int(y)), (WIDTH - m_r, int(y)), 1)
            screen.blit(font.render(str(tick), True, (130, 130, 130)), (m_l - 140, int(y) - 20))
        for i in range(len(self.data)):
            x = int(m_l + i * dx)
            if i % max(1, len(self.data) // 10) == 0:
                screen.blit(font.render(str(i + 1), True, (130, 130, 130)),
                            (x - 15, HEIGHT - m_b + 10))
        screen.blit(font.render("Generation", True, (180, 180, 180)),
                    (WIDTH // 2 - 80, HEIGHT - m_b + 55))
        screen.blit(font.render("Fitness", True, (180, 180, 180)), (10, HEIGHT // 2 - 20))
        def gy(v): return int((HEIGHT - m_b) - (v / max_val * g_h))
        for i in range(len(self.data) - 1):
            d1, d2 = self.data[i], self.data[i + 1]
            x1, x2 = int(m_l + i * dx), int(m_l + (i + 1) * dx)
            bar_h = (d2[3] / POP_SIZE) * g_h
            pygame.draw.rect(screen, (60, 0, 0), (x2 - 4, int((HEIGHT - m_b) - bar_h), 8, int(bar_h)))
            pygame.draw.line(screen, WINNER_COLOR, (x1, gy(d1[0])), (x2, gy(d2[0])), 5)
            pygame.draw.line(screen, AVG_COLOR, (x1, gy(d1[1])), (x2, gy(d2[1])), 3)
            pygame.draw.line(screen, WORST_COLOR, (x1, gy(d1[2])), (x2, gy(d2[2])), 3)
        for i, (t, c) in enumerate([("BEST", WINNER_COLOR), ("AVG", AVG_COLOR),
                                      ("WORST", WORST_COLOR), ("DEAD", (180, 0, 0))]):
            pygame.draw.rect(screen, c, (m_l + i * 520, 90, 35, 35))
            screen.blit(font.render(t, True, WHITE), (m_l + 45 + i * 520, 90))

    def draw_fitness_breakdown(self, screen, font):
        screen.fill((10, 10, 20))
        if len(self.fitness_components_log) < 2:
            msg = font.render("Not enough data.", True, WHITE)
            screen.blit(msg, (WIDTH // 2 - 200, HEIGHT // 2))
            return
        title = font.render("FITNESS BREAKDOWN — BEST WALKER PER GENERATION", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 35))
        m_l, m_r, m_t, m_b = 200, 120, 160, 160
        g_w, g_h = WIDTH - m_l - m_r, HEIGHT - m_t - m_b
        n = len(self.fitness_components_log)
        dx = g_w / max(n - 1, 1)
        series = [
            ("distance",   (80, 200, 80),   "DISTANCE (gated)"),
            ("alt",        (180, 120, 255), "ALTERNATION"),
            ("lifted",     (80, 180, 255),  "LIFTED FOOT"),
            ("upright",    (200, 255, 80),  "UPRIGHT"),
            ("torso_pen",  (255, 100, 50),  "TORSO DRAG PEN"),
            ("head_pen",   (255, 40, 40),   "HEAD DRAG PEN"),
            ("tilt_pen",   (255, 60, 100),  "TILT PEN"),
            ("slide_pen",  (255, 150, 40),  "SLIDE PEN"),
            ("energy_pen", (255, 220, 100), "ENERGY PEN"),
        ]
        all_vals = []
        for comp in self.fitness_components_log:
            for k, _, _ in series:
                all_vals.append(comp[k])
        max_val = max(all_vals + [100])
        def gy(v): return int((HEIGHT - m_b) - (max(0.0, v) / max_val * g_h))
        pygame.draw.line(screen, WHITE, (m_l, HEIGHT - m_b), (WIDTH - m_r, HEIGHT - m_b), 3)
        pygame.draw.line(screen, WHITE, (m_l, m_t), (m_l, HEIGHT - m_b), 3)
        for tick in range(0, int(max_val) + 1, max(1, int(max_val // 6))):
            y = gy(tick)
            pygame.draw.line(screen, (40, 40, 55), (m_l, y), (WIDTH - m_r, y), 1)
            screen.blit(font.render(str(tick), True, (130, 130, 130)), (m_l - 140, y - 20))
        for i in range(n):
            x = int(m_l + i * dx)
            if i % max(1, n // 10) == 0:
                screen.blit(font.render(str(i + 1), True, (130, 130, 130)), (x - 15, HEIGHT - m_b + 10))
        screen.blit(font.render("Generation", True, (180, 180, 180)),
                    (WIDTH // 2 - 80, HEIGHT - m_b + 55))
        screen.blit(font.render("Score / Penalty", True, (180, 180, 180)), (10, HEIGHT // 2 - 20))
        for key, col, _ in series:
            pts = [(int(m_l + i * dx), gy(self.fitness_components_log[i][key])) for i in range(n)]
            if len(pts) > 1:
                pygame.draw.lines(screen, col, False, pts, 2)
        for i, (_, col, lbl) in enumerate(series):
            row = i // 5
            idx = i % 5
            lx = m_l + idx * 340
            ly = 90 + row * 35
            pygame.draw.rect(screen, col, (lx, ly, 28, 14))
            screen.blit(font.render(lbl, True, WHITE), (lx + 36, ly - 4))

    def draw_oscilloscope(self, screen, font):
        screen.fill((10, 15, 10))
        if not self.brain_log:
            return
        m_l, m_t = 140, 120
        g_w = WIDTH - m_l - 80
        g_h = HEIGHT - 280
        pygame.draw.rect(screen, (15, 25, 15), (m_l, m_t, g_w, g_h))
        pygame.draw.line(screen, (0, 140, 0), (m_l, m_t + g_h // 2), (m_l + g_w, m_t + g_h // 2), 2)
        pygame.draw.rect(screen, (0, 180, 0), (m_l, m_t, g_w, g_h), 2)
        colors = [(255, 50, 50), (50, 255, 50), (50, 100, 255), (255, 255, 50),
                  (255, 50, 255), (50, 255, 255), (255, 150, 50), (150, 50, 255),
                  (255, 255, 255), (100, 100, 100)]
        labels = ["SPINE", "NECK", "L1-HIP", "L1-KNEE", "L2-HIP", "L2-KNEE",
                  "L3-HIP", "L3-KNEE", "L4-HIP", "L4-KNEE"]
        dx = g_w / self.max_log
        for m_idx in range(10):
            pts = []
            for i, frame in enumerate(self.brain_log):
                val = frame[m_idx]
                x = m_l + i * dx
                y = (m_t + g_h // 2) - (val * (g_h // 2.5))
                pts.append((x, y))
            if len(pts) > 1:
                pygame.draw.lines(screen, colors[m_idx], False, pts, 2)
        for i, (t, c) in enumerate(zip(labels, colors)):
            lx = m_l + i * (g_w // 10)
            pygame.draw.rect(screen, c, (lx, HEIGHT - 80, 20, 20))
            screen.blit(font.render(t, True, c), (lx + 26, HEIGHT - 85))
        title = font.render("LEADER BRAIN — MOTOR OSCILLOSCOPE", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 35))

    def draw_gait_signature(self, screen, font):
        screen.fill((10, 10, 15))
        cx, cy = WIDTH // 2, HEIGHT // 2
        radius = min(cx, cy) - 160
        pygame.draw.line(screen, (60, 60, 60), (cx - radius, cy), (cx + radius, cy), 2)
        pygame.draw.line(screen, (60, 60, 60), (cx, cy - radius), (cx, cy + radius), 2)
        pygame.draw.circle(screen, (30, 30, 40), (cx, cy), radius, 1)
        scale = radius * 0.8
        for i in range(1, len(self.gait_trail)):
            p1, p2 = self.gait_trail[i - 1], self.gait_trail[i]
            if p1[2] <= 0:
                continue
            alpha = p2[2] / 255.0
            color = (int(WINNER_COLOR[0] * alpha),
                     int(WINNER_COLOR[1] * alpha),
                     int(WINNER_COLOR[2] * alpha))
            start = (cx + p1[0] * scale, cy + p1[1] * scale)
            end = (cx + p2[0] * scale, cy + p2[1] * scale)
            pygame.draw.line(screen, color, start, end, 3)
        title = font.render("GAIT SIGNATURE — LEADER MOTOR ORBIT", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))

    def draw_body_evolution(self, screen, font, small_font):
        screen.fill((12, 15, 20))
        if len(self.body_log) < 2:
            msg = font.render("Not enough data.", True, WHITE)
            screen.blit(msg, (WIDTH // 2 - 200, HEIGHT // 2))
            return
        title = font.render("BODY EVOLUTION — LEADER'S PARAMS OVER GENERATIONS", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 35))
        sub = small_font.render("Normalised to starting midpoint. 1.0 = unchanged. Above = larger, below = smaller.",
                                 True, (130, 130, 150))
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 90))

        m_l, m_r, m_t, m_b = 220, 280, 140, 160
        g_w, g_h = WIDTH - m_l - m_r, HEIGHT - m_t - m_b
        n = len(self.body_log)
        dx = g_w / max(n - 1, 1)
        mid = self.body_log[0]

        plot_keys = [
            ("spine_limit",       (255, 180, 80)),
            ("upper_leg_len",     (80, 255, 120)),
            ("lower_leg_len",     (80, 220, 255)),
            ("torso_rear_mass",   (255, 80, 120)),
            ("torso_front_mass",  (200, 80, 255)),
            ("foot_friction",     (255, 255, 120)),
            ("hip_limit_fwd",     (120, 220, 180)),
            ("knee_limit",        (255, 120, 200)),
        ]

        series = {}
        for key, col in plot_keys:
            base = mid[key] if mid[key] != 0 else 1.0
            series[key] = [self.body_log[i][key] / base for i in range(n)]

        all_vals = [v for s in series.values() for v in s]
        y_min = min(all_vals + [0.5])
        y_max = max(all_vals + [1.5])
        y_range = y_max - y_min

        def gy(v): return int((HEIGHT - m_b) - ((v - y_min) / y_range * g_h))

        pygame.draw.line(screen, WHITE, (m_l, HEIGHT - m_b), (WIDTH - m_r, HEIGHT - m_b), 3)
        pygame.draw.line(screen, WHITE, (m_l, m_t), (m_l, HEIGHT - m_b), 3)

        for tick_v in [y_min, 1.0, y_max]:
            y = gy(tick_v)
            is_unity = abs(tick_v - 1.0) < 0.01
            line_col = (200, 180, 80) if is_unity else (80, 80, 80)
            line_w = 2 if is_unity else 1
            pygame.draw.line(screen, line_col, (m_l, y), (WIDTH - m_r, y), line_w)
            screen.blit(small_font.render(f"{tick_v:.2f}x", True, (160, 160, 180)),
                        (m_l - 100, y - 16))

        for i in range(n):
            x = int(m_l + i * dx)
            if i % max(1, n // 10) == 0:
                screen.blit(small_font.render(str(i + 1), True, (130, 130, 130)),
                            (x - 15, HEIGHT - m_b + 10))
        screen.blit(small_font.render("Generation", True, (180, 180, 180)),
                    (WIDTH // 2 - 80, HEIGHT - m_b + 55))

        for key, col in plot_keys:
            pts = [(int(m_l + i * dx), gy(series[key][i])) for i in range(n)]
            if len(pts) > 1:
                pygame.draw.lines(screen, col, False, pts, 3)

        for i, (key, col) in enumerate(plot_keys):
            lx = WIDTH - m_r + 10
            ly = m_t + i * 38
            pygame.draw.rect(screen, col, (lx, ly, 22, 16))
            screen.blit(small_font.render(key, True, WHITE), (lx + 30, ly - 2))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def load_midpoint_body():
    if os.path.exists("skeleton.json"):
        try:
            with open("skeleton.json") as f:
                loaded = json.load(f)
            print("Loaded skeleton.json")
            for k, v in DEFAULT_BODY.items():
                if k not in loaded:
                    loaded[k] = v
            return loaded
        except Exception as e:
            print(f"skeleton.json load failed ({e})")
    return dict(DEFAULT_BODY)


def spawn_genome(midpoint):
    body = dict(midpoint)
    for key, (fmin, fmax) in BODY_PARAM_BOUNDS.items():
        mid = midpoint[key]
        low, high = mid * fmin, mid * fmax
        noise = random.gauss(0, 0.05) * mid
        body[key] = max(low, min(high, mid + noise))
    return Genome(RNNBrain(), body, midpoint)


def save_winner(leader):
    try:
        leader.nn.save("winner_brain.npz")
        with open("winner_body.json", "w") as f:
            json.dump(leader.body_params, f, indent=2)
        return True
    except Exception as e:
        print(f"Save failed: {e}")
        return False


def load_winner_genome(midpoint):
    try:
        brain = RNNBrain.load("winner_brain.npz")
        with open("winner_body.json") as f:
            body = json.load(f)
        return Genome(brain, body, midpoint)
    except Exception as e:
        print(f"Load winner failed: {e}")
        return None


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
    clock = pygame.time.Clock()
    space = pymunk.Space()

    midpoint = load_midpoint_body()
    space.gravity = (0, midpoint.get("gravity", 2000))
    space.damping = 0.9
    space.collision_slop = 0.5
    space.collision_bias = pow(1.0 - 0.1, 60)
    floor_y = HEIGHT - 150
    history = History()
    state = STATE_SIM

    # Solid static floor box — top face at floor_y, 200 px deep, spans full sim range.
    floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    floor_verts = [(-5000, floor_y), (15000, floor_y),
                   (15000, floor_y + 200), (-5000, floor_y + 200)]
    floor = pymunk.Poly(floor_body, floor_verts)
    floor.friction = 2.0
    floor.filter = pymunk.ShapeFilter(categories=CAT_GROUND)
    space.add(floor_body, floor)

    seed_genome = load_winner_genome(midpoint)
    if seed_genome is not None:
        print("Seeding population with loaded winner")
        population = [Quadruped(space, seed_genome.clone(mutate_brain_rate=0.05)) for _ in range(POP_SIZE)]
    else:
        population = [Quadruped(space, spawn_genome(midpoint)) for _ in range(POP_SIZE)]

    global CURRENT_GEN
    gen, timer, cam_x, ff = 1, 0, 0, False
    CURRENT_GEN = gen
    flash_msg, flash_timer = "", 0

    font = pygame.font.SysFont("monospace", 42, bold=True)
    small_font = pygame.font.SysFont("monospace", 28, bold=True)
    tiny_font = pygame.font.SysFont("monospace", 22, bold=True)

    btn_w, btn_h, btn_gap = 250, 72, 12
    btn_x0 = WIDTH - btn_w - 40
    btn_y0 = 40
    def mk_btn(i):
        return pygame.Rect(btn_x0, btn_y0 + i * (btn_h + btn_gap), btn_w, btn_h)
    nav_btn      = mk_btn(0)
    brain_btn    = mk_btn(1)
    gait_btn     = mk_btn(2)
    fitness_btn  = mk_btn(3)
    body_btn     = mk_btn(4)
    ff_btn       = mk_btn(5)
    save_btn     = mk_btn(6)
    leaders_btn  = mk_btn(7)

    show_leaders_only = False

    # Pre-render static button label surfaces
    _lbl_brain   = small_font.render("BRAIN",    True, WHITE)
    _lbl_gait    = small_font.render("GAIT",     True, WHITE)
    _lbl_fitness = small_font.render("FITNESS",  True, WHITE)
    _lbl_body    = small_font.render("BODY",     True, WHITE)
    _lbl_save    = small_font.render("SAVE WIN", True, WHITE)
    _lbl_stats   = small_font.render("STATS",    True, WHITE)
    _lbl_back    = small_font.render("BACK",     True, WHITE)
    _lbl_ff      = small_font.render("FAST FWD", True, WHITE)
    _lbl_normal  = small_font.render("NORMAL",   True, WHITE)
    _lbl_leaders = small_font.render("LEADERS",  True, WHITE)

    # Sort-order cache for draw loop
    draw_order_cache = []
    draw_order_dead_count = -1

    # Do an initial niche tagging so colours appear from generation 1
    tag_creatures_by_niche(population)

    while True:
        dt = 1.0 / FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if nav_btn.collidepoint(event.pos):
                    state = STATE_SIM if state != STATE_SIM else STATE_STATS
                elif brain_btn.collidepoint(event.pos):
                    state = STATE_BRAIN if state != STATE_BRAIN else STATE_SIM
                elif gait_btn.collidepoint(event.pos):
                    state = STATE_GAIT if state != STATE_GAIT else STATE_SIM
                elif fitness_btn.collidepoint(event.pos):
                    state = STATE_FITNESS if state != STATE_FITNESS else STATE_SIM
                elif body_btn.collidepoint(event.pos):
                    state = STATE_BODY if state != STATE_BODY else STATE_SIM
                elif ff_btn.collidepoint(event.pos):
                    ff = not ff
                elif leaders_btn.collidepoint(event.pos):
                    show_leaders_only = not show_leaders_only
                elif save_btn.collidepoint(event.pos):
                    leader = max(population, key=lambda x: x.cached_fitness)
                    if save_winner(leader):
                        flash_msg = f"Winner saved (fit={int(leader.cached_fitness)})"
                    else:
                        flash_msg = "Save failed"
                    flash_timer = 240

        steps = FF_MULTIPLIER if ff else 1
        for _ in range(steps):
            timer += dt
            for q in population:
                if not q.is_dead:
                    q.update(timer, floor_y)
            for _ in range(SUBSTEPS):
                space.step(dt / SUBSTEPS)

        leader = max(population, key=lambda x: x.cached_fitness)
        history.log_brain(leader.last_motor_out)

        if state == STATE_SIM:
            cam_x += (leader.bodies[0].position.x - cam_x - 800) * 0.1
            screen.fill(SKY_COLOR)
            pygame.draw.rect(screen, GROUND_COLOR, (0, floor_y, WIDTH, HEIGHT - floor_y))
            for mx in range(int(cam_x // 400) * 400, int(cam_x) + WIDTH + 400, 400):
                sx = mx - cam_x
                pygame.draw.line(screen, (30, 30, 40), (sx, 0), (sx, floor_y), 1)
                tag = small_font.render(str(mx), True, (60, 60, 70))
                screen.blit(tag, (sx + 5, floor_y - 30))

            # Rebuild sort cache only when death count changes
            current_dead = sum(1 for q in population if q.is_dead)
            if current_dead != draw_order_dead_count or len(draw_order_cache) != len(population):
                draw_order_cache[:] = sorted(population, key=lambda x: x.is_dead, reverse=True)
                draw_order_dead_count = current_dead

            if show_leaders_only:
                alive_pop = [q for q in population if not q.is_dead]
                niche_leaders = assign_live_leaders(alive_pop)
                leader_niche_map = {}
                for niche, lq in niche_leaders.items():
                    if lq not in leader_niche_map:
                        leader_niche_map[lq] = []
                    leader_niche_map[lq].append(niche)
                for q in draw_order_cache:
                    if q.is_dead or q not in leader_niche_map:
                        continue
                    niche = getattr(q, "niche_tag", "distance") or "distance"
                    base_col = NICHE_COLORS[niche]
                    body_col = base_col if q == leader else tuple(int(c * 0.75) for c in base_col)
                    joint_col = JOINT_LEADER if q == leader else JOINT_DOT
                    q.draw(screen, cam_x, body_col, joint_col, leader=(q == leader))
                    torso_pos = q.bodies[0].position
                    lbl_x = int(torso_pos.x - cam_x)
                    lbl_y = int(torso_pos.y) - 70
                    for ni, niche_name in enumerate(leader_niche_map[q]):
                        abbrev_surf = tiny_font.render(NICHE_ABBREV[niche_name], True,
                                                       NICHE_COLORS[niche_name])
                        screen.blit(abbrev_surf, (lbl_x + ni * 55, lbl_y))
            else:
                for q in draw_order_cache:
                    if q.is_dead:
                        continue
                    niche = getattr(q, "niche_tag", "distance") or "distance"
                    base_col = NICHE_COLORS[niche]
                    if q.is_glitched:
                        body_col, joint_col = (255, 0, 255), WHITE
                    elif q == leader:
                        body_col = base_col
                        joint_col = JOINT_LEADER
                    else:
                        body_col = tuple(int(c * 0.75) for c in base_col)
                        joint_col = JOINT_DOT
                    q.draw(screen, cam_x, body_col, joint_col, leader=(q == leader))

            stage_name, stage_col = curriculum_stage(gen)
            gen_surf = font.render(f"GEN: {gen}  FIT: {int(leader.cached_fitness)}", True, WHITE)
            stage_surf = font.render(f"[{stage_name}]", True, stage_col)
            screen.blit(gen_surf, (50, 50))
            screen.blit(stage_surf, (50 + gen_surf.get_width() + 18, 50))

            # Niche HUD — shows best score in each niche this generation
            for idx, niche in enumerate(NICHES):
                col = NICHE_COLORS[niche]
                best_q = max(population, key=lambda q: niche_score(q, niche))
                score = niche_score(best_q, niche)
                pygame.draw.rect(screen, col, (50, 120 + idx * 45, 22, 22))
                lbl = small_font.render(f"{niche.upper()}: {score:.1f}", True, col)
                screen.blit(lbl, (80, 118 + idx * 45))

            # Leader stats
            lines = [
                f"TOUCHDOWNS: {leader.foot_touchdown_events}  (gate target: {TARGET_TOUCHDOWNS})",
                f"FEET FACTOR: {min(1.0, leader.foot_touchdown_events / TARGET_TOUCHDOWNS):.2f}",
                f"LIFTED frames: {leader.lifted_foot_frames}",
                f"UPRIGHT frames: {leader.upright_frames}",
                f"SLIDE frames: {leader.slide_frames}",
                f"TORSO DRAG: {leader.torso_drag_frames}  HEAD DRAG: {leader.head_drag_frames}",
            ]
            for i, l in enumerate(lines):
                screen.blit(tiny_font.render(l, True, WHITE), (50, HEIGHT - 200 + i * 28))

        elif state == STATE_STATS:
            history.draw_stats(screen, font)
        elif state == STATE_BRAIN:
            history.draw_oscilloscope(screen, small_font)
        elif state == STATE_GAIT:
            history.draw_gait_signature(screen, small_font)
        elif state == STATE_FITNESS:
            history.draw_fitness_breakdown(screen, small_font)
        elif state == STATE_BODY:
            history.draw_body_evolution(screen, font, small_font)

        button_entries = [
            (nav_btn,     _lbl_back    if state != STATE_SIM else _lbl_stats, BTN_BG),
            (brain_btn,   _lbl_brain,   BTN_BG),
            (gait_btn,    _lbl_gait,    BTN_BG),
            (fitness_btn, _lbl_fitness, BTN_BG),
            (body_btn,    _lbl_body,    BTN_BG),
            (ff_btn,      _lbl_normal  if ff else _lbl_ff,   BTN_HI if ff else BTN_BG),
            (save_btn,    _lbl_save,    (50, 80, 50)),
            (leaders_btn, _lbl_leaders, (40, 80, 40) if show_leaders_only else BTN_BG),
        ]
        for rect, lbl, col in button_entries:
            pygame.draw.rect(screen, col, rect, border_radius=10)
            screen.blit(lbl, (rect.centerx - lbl.get_width() // 2,
                              rect.centery - lbl.get_height() // 2))

        if flash_timer > 0:
            flash_timer -= 1
            msg_surf = small_font.render(flash_msg, True, (120, 255, 140))
            screen.blit(msg_surf, (WIDTH // 2 - msg_surf.get_width() // 2, HEIGHT - 80))

        # --- GENERATION TURNOVER ---
        if timer >= GEN_TIME or all(q.is_dead for q in population):
            tag_creatures_by_niche(population)
            history.record(population)
            new_genomes = build_next_generation_niche(population)
            for q in population:
                q.cleanup()
            population = [Quadruped(space, g) for g in new_genomes]
            tag_creatures_by_niche(population)
            gen += 1
            CURRENT_GEN = gen
            timer = 0
            draw_order_cache.clear()
            draw_order_dead_count = -1

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()