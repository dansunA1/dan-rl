use dan_rl_core::{
    types::F,
    traits::{
        Info,
        Env,
        Action,
        RuleActor,
    },
};

struct MyInfo {}

impl Info for MyInfo {
    const N: usize = 1;
    const KEYS: [&str; Self::N] = ["info"];
    fn write(&self, info: &mut [F;Self::N]) {
        info[0] = 0.0;
    }
}

struct MyEnv {
    info: MyInfo,
}

impl Env for MyEnv {
    const SDIM: usize = 1;
    const RDIM: usize = 1;
    type Info = MyInfo;
    fn info(&self) -> &Self::Info {
        &self.info
    }
    fn write_state(&self, state: &mut [F;Self::SDIM]) {
        state[0] = 0.0;
    }
    fn write_reward(&self, reward: &mut [F;Self::RDIM]) {
        reward[0] = 0.0;
    }
}

fn make_env(x: usize) -> MyEnv {
    MyEnv {
        info: MyInfo{},
    }
}

struct MyAction {}

impl Action<MyEnv> for MyAction {
    const DIM: usize = 1;
    fn encode(&self, out: &mut [F;Self::DIM]) {
        out[0] = 0.0;
    }
    fn decode(from: &[F;Self::DIM]) -> Self {
        MyAction {
        }
    }
    fn perform(&self, env: &mut MyEnv) -> (bool,bool) {
        (false, false)
    }
}

struct MyActorA {}

impl RuleActor<MyEnv, MyAction> for MyActorA {
    fn act(&mut self, env: &MyEnv) -> (MyAction, F) {
        (MyAction { }, 0.0)
    }
    fn reset(&mut self) {
    }
}
fn make_actor_a(x: usize) -> MyActorA {
    MyActorA {
    }
}

struct MyActorB {}

impl RuleActor<MyEnv, MyAction> for MyActorB {
    fn act(&mut self, env: &MyEnv) -> (MyAction, F) {
        (MyAction { }, 0.0)
    }
    fn reset(&mut self) {
    }
}
fn make_actor_b(x: usize, y: usize) -> MyActorB {
    MyActorB {
    }
}

dan_rl_python::make_module!(
    module_name: test_dan_rl_python,
    action: MyAction,
    env_factories: [
        MyEnv(x: usize) => make_env(x),
    ],
    actor_factories: [
        MyActorA(x: usize) => make_actor_a(x),
        MyActorB(x: usize, y: usize) => make_actor_b(x, y),
    ],
);

