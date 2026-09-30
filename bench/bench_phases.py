"""Where does one flyvis training iteration actually go? Times each phase in isolation.

Phases follow MultiTaskSolver.train's handle_batch (solver.py): data batch -> stimulus ->
network rollout -> decoder -> loss -> loss.backward(retain_graph=True) -> optimizer.step()
-> activity-penalty backward. steady_state runs once per epoch, not per iteration.
"""
import os, sys, time
sys.path.insert(0, "/content")
import bench_gpu as bg
import torch

N = int(os.environ.get("PHASE_REPS", "8"))


def t(fn, n=N, label=""):
    for _ in range(2):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    s = (time.time() - t0) / n
    print(f"RESULT phase {label} s={s:.4f}", flush=True)
    return s


def main():
    bg.make_mock_sintel()
    solver, n_batches = bg.build_solver("phases", 1)
    net, task, dec, opt = solver.network, solver.task, solver.decoder, solver.optimizer
    dt = task.dataset.dt
    print(f"[env] gpu={torch.cuda.get_device_name(0)} cpus={os.cpu_count()} arm={bg.ARM} "
          f"batch={solver.config.task.batch_size} n_frames={task.dataset.n_frames} dt={dt} "
          f"batches_per_epoch={n_batches}", flush=True)

    # 1. data: pull batches from the dataloader, augmentation included, nothing else
    def data_only():
        with task.dataset.augmentation(True):
            for i, d in enumerate(task.train_data):
                if i >= 1:
                    break
        return d
    d_data = t(data_only, label="dataloader_batch")

    with task.dataset.augmentation(True):
        data = next(iter(task.train_data))
    lum = data["lum"]
    n_samples, n_frames = lum.shape[0], lum.shape[1]

    # 2. steady state (once per epoch in flyvis)
    d_ss = t(lambda: net.steady_state(t_pre=0.5, dt=dt, batch_size=n_samples, value=0.5),
             n=3, label="steady_state_per_epoch")
    ss = net.steady_state(t_pre=0.5, dt=dt, batch_size=n_samples, value=0.5)

    def fwd():
        net.stimulus.zero(n_samples, n_frames)
        net.stimulus.add_input(lum)
        return net(net.stimulus(), dt, state=ss)

    d_fwd = t(lambda: fwd(), label="network_forward")
    act = fwd()

    def decode_loss(a):
        y_est = dec["flow"](a)
        return task.loss(y_est, data["flow"], "flow")

    d_dec = t(lambda: decode_loss(act), label="decoder_plus_loss")

    def full_backward():
        opt.zero_grad()
        a = fwd()
        loss = decode_loss(a)
        loss.backward(retain_graph=True)
        return a

    d_all = t(full_backward, label="fwd_decode_loss_backward")

    def with_step_and_penalty():
        opt.zero_grad()
        a = fwd()
        loss = decode_loss(a)
        loss.backward(retain_graph=True)
        opt.step()
        solver.penalty(activity=a, iteration=0)
    d_iter = t(with_step_and_penalty, label="full_iteration")

    print(f"RESULT phase SUMMARY per_iteration_s={d_iter:.4f} "
          f"data={d_data:.4f} forward={d_fwd:.4f} decoder_loss={d_dec:.4f} "
          f"backward={max(d_all - d_fwd - d_dec, 0):.4f} step_and_penalty={max(d_iter - d_all, 0):.4f} "
          f"steady_state_per_epoch={d_ss:.4f} amortized_ss={d_ss / max(n_batches, 1):.4f}", flush=True)
    print("PHASES DONE", flush=True)


if __name__ == "__main__":
    main()
