### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ d1134e40-2180-4890-bdf3-99d422a6d625
using Pkg

# ╔═╡ 383c70ae-bc93-4b50-b905-6ba5baf16875
Pkg.activate("Assignment2.toml")

# ╔═╡ 91a2317f-f790-4cf6-a74c-265209b4c985
Pkg.add("FFTW")

# ╔═╡ e6dc4420-c615-11eb-26e4-69e5c918e2d6
using PlutoUI

# ╔═╡ 5b419524-0372-4b7f-839b-77d513c0896e
using Random

# ╔═╡ 418e3a78-dccd-4d25-9378-bbf766a6965a
using Flux

# ╔═╡ 081d80b0-cfba-11eb-3ca8-bb05ffe71b5b
using Images

# ╔═╡ 0af4bc90-cfba-11eb-2643-c3f13710773b
using FFTW

# ╔═╡ 8f55e841-ea0c-4fd2-9671-6a133b9233b1
using Logging: with_logger, global_logger

# ╔═╡ 71d85a6c-3269-4961-8995-5a1706d1c80c
import Base.-

# ╔═╡ 1d1b7b00-d6e6-45b2-a61c-10d4b5482b1a
cnt = 0

# ╔═╡ 8e63bad2-60fe-4a7a-9a39-11281bd2be45
PATH_NORMAL = "C:\\Users\\Mino\\Desktop\\Assignment\\chest_xray"

# ╔═╡ 65add21c-7a73-4908-9653-1ee1783cd003
TARGET_NORMAL_FILE = "Dataset/Normal"

# ╔═╡ be6e1bfa-896e-4c02-a395-bfc60b73ee9b
PATH_PNEUMONIA = "C:\\Users\\Mino\\Desktop\\Assignment\\chest_xray\\train\\PNEUMONIA"

# ╔═╡ 74b262e9-b341-4eeb-a23b-620b74ec9802
TARGET_PNEUMONIA_FILE = "Dataset/Pneumonia"

# ╔═╡ 70791f0e-1689-4eb5-a85d-3103f1ba01d2
struct image_name
	nn
    x0
end

# ╔═╡ a949b351-e202-42be-be47-3f7a5fdf9f5a
image_names = readdir(PATH_NORMAL)

# ╔═╡ 4371d9cc-5a20-40f3-9619-9297bd223eae
image_names2 = readdir(PATH_PNEUMONIA)

# ╔═╡ 8242f7a4-7051-43e3-bc4c-9e7cb2472c13
image_names

# ╔═╡ 7dfa838e-36fa-426a-8be7-99ca7eedf1b5
image_names2

# ╔═╡ 1736bd60-6ef2-429c-9a30-f7f29123f202
Random.shuffle(image_names)

# ╔═╡ 23f3ef40-c432-4387-b493-0f13443e23ef
struct LeNet5Iterator
    nn
    x0
end

# ╔═╡ 09f99f53-0e3a-4675-81c6-42268c424dc5
function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            x -> reshape(x, imgsize..., :),
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

# ╔═╡ fc447c16-39f5-44ea-b59e-5e9dbe5f829b
function (nn::LeNet5)(x)
    # check input
    if ndims(x) == 2
        w, h = size(x)
        x = reshape(x, (w, h, 1, 1))
    elseif ndims(x) == 3
        w, h, n = size(x)
        x = reshape(x, (w, h, 1, n))
    end
    @assert ndims(x) == 4
    LeNet5Iterator(nn, x)
end

# ╔═╡ 9898db48-9a27-4372-8603-51eb944fa3dc
Base.iterate(it::LeNet5Iterator, ::Nothing) = nothing

# ╔═╡ b10ff944-a73d-41b5-9e1b-bc6850eaf300
function Base.iterate(it::LeNet5Iterator)
    Base.iterate(it, (it.x0, :conv1))
end

# ╔═╡ 0ef61f67-0fd6-473d-ac48-3668ad244882
Base.IteratorSize(it::LeNet5Iterator) = Base.HasLength()

# ╔═╡ cec12129-f65c-4fe6-9133-279840dd4c43
Base.length(it::LeNet5Iterator) = 8

# ╔═╡ 7051aea1-a43c-4d46-a79c-f8fae42fb6b5
Base.last(it::LeNet5Iterator) = Iterators.drop(it,7) |> collect |> last |> last

# ╔═╡ d7047c6d-f9ee-476e-8aae-d2c0906fc4ae
function accuracy(x,y)
     return mean(onecold(model(x)) .== onecold(y))
 end

# ╔═╡ cb04fff3-fe4e-4939-aa8a-4702389cbe0c
function training_model(loss = loss, acmodel = model, train_set = train_set, test_set = test_set, opt = ADAM(0.001),
    save_name = "SEM_convnet.bson")
    best_acc = 0.0
    last_improvement = 0
    training_losses = Vector{Float32}()
    test_losses = Vector{Float32}()
    accuracies = Vector{Float32}()
    for epoch_idx in 1:100

# ╔═╡ 6d504d14-8f0e-46c2-b0a8-81ff642b700b
 @eval Flux.istraining() = true
        Flux.train!(loss, params(model), train_set, opt)
        training_loss = sum(loss(train_set[i]...) for i in 1:length(train_set))
        @eval Flux.istraining() = false
        test_loss = loss(test_set...)
        acc = accuracy(test_set...)

# ╔═╡ a6c38198-7a91-437d-a34c-59212acfdb99
println("Epoch ", epoch_idx,": Training Loss = ", training_loss, ", Test accuracy = ", acc)
        append!(training_losses, training_loss)
        append!(accuracies, acc)
        append!(test_losses, test_loss)

# ╔═╡ 982010c0-195e-4621-96a9-7c7bb636e852
if acc >= 0.999
            break
        end

# ╔═╡ 7cf8003f-f425-4b9b-90fe-d2b8a8a66219
if acc > best_acc
            println("New best accuracy")
            save_name_best = "bson_outputs/"split(save_name, ".")[1]"_best.bson"
            BSON.@save save_name_best model epoch_idx accuracies training_loss test_loss
            best_acc = acc
            last_improvement = epoch_idx
        end

# ╔═╡ 65f2ed5e-f359-458d-8a16-6f13b033667e


# ╔═╡ 511077ed-537b-40ee-ab9f-cb908d7420c6
		for i in range(142)
	
	range(;length = 142, start = 1)
1:142

range(;length = 142, start = 1, stop = 150)
1.0:142.0:150.0

range(; start=142, step = 2, length=200)
142:4:401
	
	
			image_name = image_names[i]
			image_path = (joinpath(KAGGLE_FILE_PATH_NORMAL,image_name))
			target_path = (joinpath(TARGET_NORMAL_DIR,image_name))
	
			cp(src::image_path,dst::target_path)
	
	range([start, stop, length]; start, stop, length, step=1)
	_range(Int64, Nothing, Nothing, Nothing)
	
			println("copying image",i)
end


# ╔═╡ 22798973-2d44-4242-9649-fed3538566ae


# ╔═╡ 4c667ef6-deff-401e-ae75-1c02a7564864


# ╔═╡ 08295b0a-fbcb-4804-8d30-b1dfb4b0666e


# ╔═╡ 3569f3c9-bf01-4488-8974-6a7e018ca4f6


# ╔═╡ 4b889069-5f8e-4413-b276-385605ae7d07


# ╔═╡ Cell order:
# ╠═e6dc4420-c615-11eb-26e4-69e5c918e2d6
# ╠═d1134e40-2180-4890-bdf3-99d422a6d625
# ╠═383c70ae-bc93-4b50-b905-6ba5baf16875
# ╠═91a2317f-f790-4cf6-a74c-265209b4c985
# ╠═5b419524-0372-4b7f-839b-77d513c0896e
# ╠═418e3a78-dccd-4d25-9378-bbf766a6965a
# ╠═081d80b0-cfba-11eb-3ca8-bb05ffe71b5b
# ╠═0af4bc90-cfba-11eb-2643-c3f13710773b
# ╠═8f55e841-ea0c-4fd2-9671-6a133b9233b1
# ╠═71d85a6c-3269-4961-8995-5a1706d1c80c
# ╠═1d1b7b00-d6e6-45b2-a61c-10d4b5482b1a
# ╠═8e63bad2-60fe-4a7a-9a39-11281bd2be45
# ╠═65add21c-7a73-4908-9653-1ee1783cd003
# ╠═be6e1bfa-896e-4c02-a395-bfc60b73ee9b
# ╠═74b262e9-b341-4eeb-a23b-620b74ec9802
# ╠═70791f0e-1689-4eb5-a85d-3103f1ba01d2
# ╠═a949b351-e202-42be-be47-3f7a5fdf9f5a
# ╠═4371d9cc-5a20-40f3-9619-9297bd223eae
# ╠═8242f7a4-7051-43e3-bc4c-9e7cb2472c13
# ╠═7dfa838e-36fa-426a-8be7-99ca7eedf1b5
# ╠═1736bd60-6ef2-429c-9a30-f7f29123f202
# ╠═23f3ef40-c432-4387-b493-0f13443e23ef
# ╠═09f99f53-0e3a-4675-81c6-42268c424dc5
# ╠═fc447c16-39f5-44ea-b59e-5e9dbe5f829b
# ╠═b10ff944-a73d-41b5-9e1b-bc6850eaf300
# ╠═9898db48-9a27-4372-8603-51eb944fa3dc
# ╠═0ef61f67-0fd6-473d-ac48-3668ad244882
# ╠═cec12129-f65c-4fe6-9133-279840dd4c43
# ╠═7051aea1-a43c-4d46-a79c-f8fae42fb6b5
# ╠═d7047c6d-f9ee-476e-8aae-d2c0906fc4ae
# ╠═cb04fff3-fe4e-4939-aa8a-4702389cbe0c
# ╠═6d504d14-8f0e-46c2-b0a8-81ff642b700b
# ╠═a6c38198-7a91-437d-a34c-59212acfdb99
# ╠═982010c0-195e-4621-96a9-7c7bb636e852
# ╠═7cf8003f-f425-4b9b-90fe-d2b8a8a66219
# ╠═65f2ed5e-f359-458d-8a16-6f13b033667e
# ╠═511077ed-537b-40ee-ab9f-cb908d7420c6
# ╠═22798973-2d44-4242-9649-fed3538566ae
# ╠═4c667ef6-deff-401e-ae75-1c02a7564864
# ╠═08295b0a-fbcb-4804-8d30-b1dfb4b0666e
# ╠═3569f3c9-bf01-4488-8974-6a7e018ca4f6
# ╠═4b889069-5f8e-4413-b276-385605ae7d07
